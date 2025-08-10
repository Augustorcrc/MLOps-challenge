# MLOps Challenge

## Descripción del Proyecto

 El proyecto incluye una API de versículos bíblicos con capacidades de monitoreo, observabilidad y escalabilidad. Completar

## Estructura del Proyecto

```
MLOps-challenge/
├── README.md                    # Documentación general del proyecto
├── requirements.txt             # Dependencias completas (desarrollo + producción)
├── .gitignore                  # Archivos a ignorar en Git
├── API/                        # Implementaciones de la API
│   ├── api_v0/                 # Versión inicial (legacy)
│   └── api_v1/                 # VERSIÓN DE ENTREGA
└── .git/                       # Control de versiones Git
```

## Tecnologías Utilizadas

- **Backend**: FastAPI, Python 3.11
- **Servidor**: Gunicorn
- **Base de Datos**: SQLite (demo), PostgreSQL (producción)
- **Monitoreo**: Prometheus, Grafana
- **Testing**: Locust para pruebas de carga
- **Containerización**: Docker, Docker Compose
- **Observabilidad**: Métricas Prometheus, logging estructurado

## Características Principales

- API RESTful para consulta de versículos bíblicos
- Sistema de cache inteligente
- Métricas de performance en tiempo real
- Logging estructurado con correlación de requests
- Health checks y readiness probes
- Configuración para desarrollo y producción
- Pruebas de carga automatizadas

## Inicio Rápido

### Opción 1: Docker Compose Completo
```bash
cd MLOps-challenge/API/api_v1
docker compose up -d
```

### Opción 2: Solo la API
```bash
cd MLOps-challenge/API/api_v1
docker compose -f docker-compose-api-only.yml up -d
```

### Opción 3: Desarrollo Local
```bash
cd MLOps-challenge
pip install -r requirements.txt
cd API/api_v1
gunicorn app.main:app -c gunicorn_conf.py
```

## Servicios Disponibles

- **API**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

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

## Limitaciones Conocidas

- **SQLite**: No ideal para alta concurrencia (comportamiento esperado en demo)
- **Performance**: Degradamiento gradual con 50+ usuarios simultáneos
- **Escalabilidad**: Plan de migración a PostgreSQL para producción

## Próximas Mejoras

- Migración a PostgreSQL
- Cache distribuido con Redis
- Rate limiting y autenticación
- Tests unitarios completos
- Pipeline CI/CD
- Despliegue en Kubernetes

## Contribución

1. Fork del repositorio
2. Crear feature branch
3. Implementar cambios con tests
4. Crear Pull Request

## Licencia

Este proyecto es parte del MLOps Challenge y está destinado para fines educativos y de demostración.

## Documentación Detallada

Para información detallada sobre la API, consulta [API/README_api.md](API/README_api.md)
