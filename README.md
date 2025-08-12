# MLOps Challenge – API y Modelo de Predicción de Churn

Este repositorio contiene la solución completa a un desafío técnico orientado al rol de **MLOps Engineer**, abarcando tanto el desarrollo de una API de alto rendimiento como la implementación de un modelo de Machine Learning para predicción de churn.

El proyecto se divide en dos componentes principales:

1. **API Bible Verses** – Servicio REST construido con FastAPI que permite consultar versículos bíblicos desde una API pública y almacenarlos en caché local para mejorar tiempos de respuesta. Incluye monitoreo con Prometheus y Grafana, pruebas de carga con Locust y una arquitectura lista para escalar en entornos productivos.

2. **Modelo de Predicción de Churn** – Pipeline completo de procesamiento de datos, entrenamiento y evaluación de un modelo de clasificación basado en XGBoost, optimizado para maximizar el recall y detectar clientes con alto riesgo de baja. Se incluye análisis exploratorio, ingeniería de características, comparación de algoritmos y sistema de caché para optimizar la preparación de datos.


Este repositorio está diseñado para mostrar no solo la implementación técnica, sino también el enfoque de **escalabilidad, observabilidad y mantenibilidad** que caracteriza a soluciones de MLOps listas para producción.


## Estructura del Proyecto

```
MLOps-challenge/
├── README.md                    # Documentación general del proyecto
├── requirements.txt             # Dependencias completas (desarrollo + producción)
├── .gitignore                  # Archivos a ignorar en Git
├── API/                        # Implementaciones de la API
│   ├── api_v0/                 # Versión inicial (legacy)
│   └── api_v1/                 # VERSIÓN DE ENTREGA
├── Data-Science                # Modelo de churn
└── .git/                       # Control de versiones Git
```

---

##  Tecnologías Utilizadas

- **Lenguaje**: Python 3.11
- **Framework API**: FastAPI
- **Machine Learning**: XGBoost, Scikit-learn, Pandas, Numpy
- **Servidor API**: Gunicorn
- **Base de Datos**: SQLite (demo) / PostgreSQL (producción)
- **Monitoreo**: Prometheus, Grafana
- **Testing de Carga**: Locust
- **Containerización**: Docker, Docker Compose
- **Observabilidad**: Logging estructurado, métricas Prometheus

---

##  Características Principales

### API Bible Verses
- API RESTful para consulta de versículos bíblicos
- Sistema de caché persistente para mejorar tiempos de respuesta
- Métricas de performance en tiempo real
- Logging estructurado con correlación de requests
- Health checks y readiness probes
- Configuración lista para entornos de desarrollo y producción
- Pruebas de carga con Locust

### Modelo de Predicción de Churn
- Pipeline reproducible con logs y sistema de caché para procesamiento geográfico
- Análisis exploratorio y manejo de desbalance de clases
- Ingeniería de características y agregación inteligente por cliente
- Comparación de múltiples algoritmos de clasificación
- Modelo final optimizado con XGBoost priorizando recall
- Métricas de evaluación y salidas con prediccion

---

##  Guías Rápidas

Cada componente cuenta con su **propia guía de ejecución** y **dependencias**:

- **API**: Instrucciones detalladas en [`API/README_api.md`](API/README_api.md)
- **Modelo de Churn**: Instrucciones detalladas en [`Data-Science/README_ds.md`](Data-Science/README_ds.md)

---

##  Licencia

Este proyecto es parte de un desafio y está destinado para fines educativos y de demostración.
