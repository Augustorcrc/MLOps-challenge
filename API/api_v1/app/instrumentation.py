from prometheus_client import Counter, Histogram, Gauge

# Contador total de requests por endpoint y método
REQUESTS = Counter("api_requests_total","Total de requests",["endpoint","method"])
# Contador de errores por endpoint, método y código de estado
ERRORS = Counter("api_request_errors_total","Total de errores",["endpoint","method","status_code"])
# Histograma de latencia por endpoint y método (P50, P95, P99)
LATENCY = Histogram("api_request_latency_seconds","Latencia por endpoint",["endpoint","method"],
                    buckets=(0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0))
# Gauge de requests siendo procesadas simultáneamente
INFLIGHT = Gauge("api_inflight_requests","Requests en vuelo")
# Contador de llamadas a APIs externas (bible-api.com)
UPSTREAM = Counter("api_upstream_calls_total","Llamadas a bible-api.com",["status"])

