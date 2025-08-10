import json
import time
import uuid
from typing import Callable

import httpx
from urllib.parse import quote
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.schemas import VerseRequest
from app import db
from app.logging_conf import get_logger, bind_request_id, get_request_id
from app.instrumentation import REQUESTS, ERRORS, LATENCY, INFLIGHT, UPSTREAM

BIBLE_API_BASE = "https://bible-api.com"

app = FastAPI(title="Bible Verses API", version="0.1.0")
log = get_logger("api")

# Init DB
db.init_db()

########################################################
# Endpoints principales 
########################################################


@app.post("/verse")
async def get_or_fetch_verse(payload: VerseRequest):
    t_start = time.perf_counter()
    ref_raw = payload.reference
    ref_norm = db.norm_ref(ref_raw)

    # cache hit?
    row = db.get_verse(ref_norm)
    if row:
        # hit: incrementa contador y devuelve
        db.insert_or_update(ref_norm, ref_raw, row[1], is_new=False)
        ref_raw_last, data_json, count = row
        s = json.loads(data_json)
        dur = time.perf_counter() - t_start
        log.info({
            "event": "cache_hit",
            "ref": ref_raw_last,
            "request_id": get_request_id(),
            "latency_ms": round(dur * 1000, 2)
        })
        return s['reference'], s['text']

    # miss: llamar a bible-api.com (normaliza para URL)
    ref_url = quote(ref_norm, safe=":")
    url = f"{BIBLE_API_BASE}/{ref_url}"
    timeout = httpx.Timeout(5.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)

    status_bucket = "success" if resp.status_code == 200 else "error"
    UPSTREAM.labels(status=status_bucket).inc()

    if resp.status_code != 200:
        # bible-api returns 404 for invalid references
        log.warning({
            "event": "upstream_error",
            "status": resp.status_code,
            "url": url,
            "request_id": get_request_id(),
        })
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Versiculo no encontrado. Verifique la referencia sea libro+capitulo+versiculo (e.g., 'John 3:16').")
        raise HTTPException(status_code=502, detail="Upstream bible-api error")

    # validar JSON antes de almacenar
    data_json = resp.text
    try:
        data_obj = json.loads(data_json)
    except json.JSONDecodeError:
        log.warning({
            "event": "upstream_invalid_json",
            "url": url,
            "request_id": get_request_id(),
        })
        raise HTTPException(status_code=502, detail="Invalid response from bible-api.")
    # guardar en cache con request_count=1
    db.insert_or_update(ref_norm, ref_raw, data_json, is_new=True)
    dur = time.perf_counter() - t_start
    log.info({
        "event": "cache_store",
        "ref": ref_raw,
        "request_id": get_request_id(),
        "latency_ms": round(dur * 1000, 2)
    })
    return data_obj['reference'], data_obj['text']


@app.get("/top")
async def get_top(n: int = 3):
    # Limitar n a máximo 10 para evitar sobrecarga
    n = min(n, 10)
    rows = db.top_n(n)
    # devuelve lista de {reference, request_count, verse (subconjunto)}
    out = []
    for ref, data_json, count in rows:
        try:
            d = json.loads(data_json)
            out.append({
                "reference": ref,
                "request_count": count,
                "text": d.get("text"),
                # "translation_name": d.get("translation_name"),
            })
        except Exception:
            out.append({"reference": ref, "request_count": count, "text": None})
    if out != []: 
        return {"top": out}
    else: 
        return JSONResponse(content="No hay registro de ninguna busqueda")
    
# Ejemplo curl http://localhost:8080/top?n=1
# Ejmplo curl -X POST "http://localhost:8080/verse" -H "Content-Type: application/json" -d "{\"reference\":\"John 3:18\"}"




########################################################
# Endpoints de observabilidad y monitoreo
########################################################

@app.middleware("http")
async def obs_middleware(request: Request, call_next: Callable):
    """
    Middleware de observabilidad que captura métricas y trazabilidad para cada request.
    
    Funcionalidades:
    1. Correlation ID: Extrae o genera X-Request-ID para trazabilidad
    2. Métricas Prometheus: Cuenta requests, errores, latencia y requests en vuelo
    3. Logging estructurado: Todos los logs de la request usan el mismo ID
    4. Headers de respuesta: Devuelve el X-Request-ID para correlación cliente-servidor
    """
    # 1. CORRELATION ID - Para trazabilidad entre logs y métricas
    # Extrae X-Request-ID del header o genera uno nuevo si no existe
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    # Guarda el ID en el contexto para que todos los logs de esta request lo usen
    bind_request_id(rid)
    
    # Extrae información de la request para etiquetar métricas
    endpoint, method = request.url.path, request.method
    
    # 2. MÉTRICAS DE PROMETHEUS - Contadores y observaciones
    # Incrementa contador de requests totales (por endpoint y método)
    REQUESTS.labels(endpoint=endpoint, method=method).inc()
    
    # Inicia medición de latencia y cuenta requests en vuelo
    t0 = time.perf_counter()
    INFLIGHT.inc()  # Gauge: requests siendo procesadas simultáneamente
    
    try:
        # Procesa la request (llama al endpoint real)
        resp: Response = await call_next(request)
        
        # 3. TRACKING DE ERRORES HTTP - Cuenta errores 4xx/5xx
        if resp.status_code >= 400:
            ERRORS.labels(endpoint=endpoint, method=method, status_code=str(resp.status_code)).inc()
        
        # 4. MEDICIÓN DE LATENCIA - Histograma para percentiles (P50, P95, P99)
        dur = time.perf_counter() - t0
        LATENCY.labels(endpoint=endpoint, method=method).observe(dur)
        
        # 5. LOG DE REQUEST COMPLETADA - Incluye latencia y status
        log.info({
            "event": "request_completed",
            "endpoint": endpoint,
            "method": method,
            "status_code": resp.status_code,
            "latency_ms": round(dur * 1000, 2),
            "request_id": rid
        })
        
        # 6. HEADER DE RESPUESTA - Cliente puede correlacionar logs
        resp.headers["X-Request-ID"] = rid
        return resp
        
    except Exception:
        # 6. MANEJO DE EXCEPCIONES NO CAPTURADAS - Errores 500
        ERRORS.labels(endpoint=endpoint, method=method, status_code="500").inc()
        log.exception("unhandled_exception")  # Stack trace completo
        raise  # Re-lanza la excepción para que FastAPI la maneje
    finally:
        # 7. LIMPIEZA - Siempre decrementa requests en vuelo
        INFLIGHT.dec()


@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"


@app.get("/ready", response_class=PlainTextResponse)
async def ready():
    return "ready"


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


