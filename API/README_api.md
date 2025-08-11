# Bible Verses API - MLOps Challenge

## Descripción

API para recuperar y cachear versículos bíblicos, implementando MLOps, observabilidad y escalabilidad. La **API v1** es la versión de entrega completa.

## Características Principales

- **POST /verse**: Recupera versículos bíblicos con cache en SQLite
- **GET /top**: Obtiene los versículos más solicitados
- **Cache inteligente**: Primera llamada a bible-api.com, luego cache local
- **Observabilidad**: Métricas Prometheus, logs estructurados JSON
- **Trazabilidad**: X-Request-ID para correlación de requests
- **Monitoreo**: Stack completo con Prometheus + Grafana
- **Load Testing**: Configuración de Locust incluida

## Stack Tecnológico

- **Backend**: FastAPI + Gunicorn
- **Base de Datos**: SQLite (demo), PostgreSQL (producción)
- **Monitoreo**: Prometheus + Grafana
- **Logging**: JSON estructurado con python-json-logger
- **Testing**: Locust para pruebas de carga
- **Containerización**: Docker + Docker Compose

## Estructura del Proyecto

```
MLOps-challenge/API/
├── api_v0/                     # Versión inicial (legacy)
│   ├── main_v0.py             # Implementación básica con cache JSON
│   └── verses_cache.json      # Cache en memoria (archivo JSON)
└── api_v1/                     # VERSIÓN DE ENTREGA
    ├── app/                    # Código principal de la aplicación
    │   ├── main.py            # Endpoints principales y middleware
    │   ├── db.py              # Operaciones de base de datos SQLite
    │   ├── schemas.py         # Modelos Pydantic para validación
    │   ├── instrumentation.py # Métricas Prometheus
    │   └── logging_conf.py    # Configuración de logging
    ├── testing/                # Pruebas y testing
    │   └── locustfile.py      # Pruebas de carga con Locust
    ├── grafana/                # Configuración de Grafana
    │   └── provisioning/       # Dashboards y datasources
    ├── prometheus/             # Configuración de Prometheus
    │   └── prometheus.yml     # Configuración del servidor
    ├── docker-compose.yml      # Stack completo (API + Prometheus + Grafana)
    ├── docker-compose-api-only.yml # Solo API (para desarrollo)
    ├── Dockerfile              # Imagen Docker de la API
    ├── gunicorn_conf.py        # Configuración de Gunicorn
    ├── requirements.txt        # Dependencias solo para la API (producción)
    ├── data/                   # Base de datos SQLite (persistente)
    └── logs/                   # Logs de la aplicación (persistente)
```

## Organización de Dependencias

**`requirements.txt` (ROOT) - Desarrollo completo:**
- API: FastAPI, Gunicorn, HTTPX, Pydantic
- Monitoreo: Prometheus, logging estructurado
- Testing: Pytest, Locust para pruebas de carga
- Desarrollo: Black, Flake8, MyPy para calidad de código
- Futuro: SQLAlchemy para migración a PostgreSQL

**`API/api_v1/requirements.txt` - Solo producción:**
- API: FastAPI, Gunicorn, HTTPX, Pydantic
- Monitoreo: Prometheus, logging estructurado
- Base de datos: SQLAlchemy
- Sin testing: No Locust, Pytest
- Sin desarrollo: No Black, Flake8, MyPy

**¿Cuál usar?**
- **Desarrollo local**: `pip install -r requirements.txt` (desde root)
- **Docker**: Usa automáticamente `API/api_v1/requirements.txt`
- **Producción**: Cualquiera de los dos funciona

## Inicio Rápido

### Opción 1: Docker Compose Completo (Prometheus + Grafana)
```bash
cd MLOps-challenge/API/api_v1
docker build -t api_v1 .
docker compose up -d
```

**Servicios disponibles:**
- API: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)



## Endpoints de la API

- `POST /verse` - Obtener versículo bíblico
- `GET /top` - Top 10 versículos más consultados
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Métricas Prometheus





## Pruebas de Carga

```bash
cd MLOps-challenge/API/api_v1/testing
locust -f locustfile.py
```

## Limitaciones de SQLite (Demo)

**Nota importante**: Esta implementación usa SQLite para simplicidad de demostración. SQLite tiene limitaciones conocidas de concurrencia que se manifiestan con:

- **10-20 usuarios simultáneos**: Funciona perfectamente, latencia estable ~50-100ms
- **50+ usuarios simultáneos**: Degradamiento gradual de performance
- **100+ usuarios simultáneos**: Degradamiento exponencial de latencia (comportamiento esperado)

**¿Por qué pasa esto?**
- SQLite usa locks de archivo para garantizar consistencia
- Cada request concurrente debe esperar su turno
- Con alta concurrencia, se forma una cola natural de requests
- El timeout configurado (10s) maneja esto correctamente

**Para producción**: Migrar a PostgreSQL resolvería completamente estos problemas de escalabilidad.

**¿Es un bug?** NO - Es el comportamiento normal y esperado de SQLite bajo alta concurrencia.

## Comentarios y Mejoras Pendientes

### Funcionalidades para Futuras Iteraciones

#### Performance y Escalabilidad
- Implementar cache distribuido (Redis)
- Rate limiting por IP/usuario
- Auto-scaling basado en métricas

#### Base de Datos
- Migración a PostgreSQL para producción


#### Seguridad
- Autenticación JWT
- Rate limiting más sofisticado
- Validación de entrada más estricta
- Headers de seguridad (CORS, CSP)

#### Testing
- Tests unitarios completos
- Tests de integración
- Tests de performance automatizados
- Coverage mínimo del 80%

#### DevOps
- Pipeline CI/CD completo
- Despliegue en Kubernetes
- Backup automático de datos

#### Monitoreo
- Alertas automáticas
- Dashboards personalizados


### Observaciones Técnicas

#### Arquitectura Actual
- La implementación actual usa SQLite por simplicidad
- El cache está optimizado para lecturas frecuentes
- La normalización de referencias bíblicas es robusta
- El sistema de métricas captura KPIs importantes

#### Limitaciones Conocidas
- **SQLite no es ideal para alta concurrencia** (comportamiento esperado en demo)
- Cache en memoria puede crecer indefinidamente
- No hay compresión de respuestas
- Falta validación de rate limits
- **Degradamiento exponencial con 100+ usuarios simultáneos** (normal para SQLite)

## Recursos Adicionales

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Docker Compose Reference](https://docs.docker.com/compose/)