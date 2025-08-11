# Challenge MLOps - Modelo de Predicción de Churn

Este proyecto implementa un modelo de machine learning para predecir la probabilidad de churn de clientes basándose en datos de atención al cliente y ubicación geográfica.

## Estructura del Proyecto

```
MLOps-challenge/Data-Science/
├── data/                           # Carpeta con los datasets
│   ├── dataset_churn_challenge.csv
│   ├── dataset_churn_zona_challenge.csv
│   └── coordenadas_procesadas_cache.csv  # Caché de coordenadas procesadas
├── src/                            # Carpeta source
│   ├── eda.ipynb                   # EDA del dataset (opcional)
│   ├── utils.py                    # Funciones auxiliares y lógica del modelo
├   └──  main.py                    # Script principal de ejecución
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
├── test/                           # Carpeta con archivos de prueba
│   ├── test_cache.py               # Pruebas del sistema de caché
│   ├── test_predicciones.py        # Pruebas de la función de predicciones
│   └── test_aggregation.py         # Pruebas de la lógica de agregación
├── logs/                           # Logs
└── results/                        # Carpeta de resultados (se crea automáticamente)
    ├── predictions.csv             # Dataset con predicciones
    └── correlation_matrix.png      # Matriz de correlación
```

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar el modelo completo:

```bash
python main.py
```

## Funcionalidades

### utils.py
Contiene todas las funciones auxiliares:

#### Procesamiento de Datos
- **Carga de datos**: Lectura de datasets CSV desde la carpeta `data/`
- **Procesamiento geográfico**: Asignación de zonas basada en coordenadas con sistema de caché
- **Feature engineering**: 
  - Codificación ordinal de `segmento_cliente` con manejo de valores desconocidos
  - One-hot encoding para `tipo_asistencia` y `zona_asignada`
  - Procesamiento de fechas y cálculo de duración de atención
- **Agregación por cliente**: 
  - Conteo de ocurrencias por `cliente_id`
  - Consolidación a un registro por cliente
  - Agregación inteligente: modo para categóricas, promedio para numéricas, primer valor para el resto
  - Lógica especial para `churn`: True si aparece algún True, False si todas son False

#### Análisis y Modelado
- **Análisis exploratorio**: Distribución del target, detección de outliers, matriz de correlación
- **Entrenamiento**: Implementación del modelo XGBoost Classifier 
- **Evaluación**: Métricas de rendimiento (Accuracy, Balanced Accuracy, ROC AUC)
- **Predicciones**: Generación de DataFrame con predicciones y métricas de acierto

#### Sistema de Caché
- **Archivo de caché**: `data/coordenadas_procesadas_cache.csv`
- **Función**: Evita reprocesar las coordenadas geográficas en ejecuciones posteriores
- **Validación**: Verifica integridad del caché comparando número de registros
- **Fallback**: Reprocesamiento automático si el caché no es válido

### main.py
Script principal que orquesta todo el proceso:
1. Configuración de logging
2. Carga de datos
3. Procesamiento y preparación
4. Entrenamiento del modelo
5. Generación de predicciones
6. Guardado de resultados

## Output

El modelo genera:

### predictions.csv
Dataset completo con:
- Todas las columnas originales del dataset procesado
- Predicciones del modelo para registros de test
- Columna `dataset_split` que indica si el registro se usó para train o test
- Una entrada por `cliente_id` con datos agregados

### correlation_matrix.png
Matriz de correlación entre variables numéricas del dataset

### metricas.csv
Metricas del modelo

## Características del Modelo

- **Algoritmo**: XGBoost Classifier
- **Manejo de desbalance**: Class weights balanceados
- **Validación**: Split estratificado 80/20 train/test
- **Métricas**: Accuracy, Balanced Accuracy, ROC AUC, Recall, F1-Score
- **Robustez**: 
  - Manejo de outliers mediante método IQR
  - Imputación de valores faltantes
  - Manejo de errores de datos y coordenadas
  - Sistema de caché para optimización de rendimiento

## Sistema de Caché

El proyecto implementa un sistema de caché inteligente para optimizar el rendimiento:

- **Archivo de caché**: `data/coordenadas_procesadas_cache.csv`
- **Función**: Evita reprocesar las coordenadas geográficas en ejecuciones posteriores
- **Validación**: Verifica que el caché tenga el mismo número de registros que el dataset original
- **Fallback**: Si el caché no es válido, reprocesa automáticamente los datos
- **Beneficios**: Reduce significativamente el tiempo de ejecución en corridas posteriores

## Logs

El sistema genera logs detallados para monitorear el proceso de entrenamiento y detectar posibles errores. Los logs incluyen información sobre:
- Procesamiento de coordenadas y asignación de zonas
- Procesamiento de fechas y cálculo de duración de visitas de clientes
- Codificación de variables categóricas
- Agregación de datos por cliente
- Entrenamiento del modelo y métricas de rendimiento
- Generación de predicciones y guardado de resultados

## Dependencias

Las principales dependencias incluyen:
- **Procesamiento de datos**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm, catboost
- **Visualización**: matplotlib, seaborn, plotly
- **Análisis geográfico**: Funciones personalizadas para procesamiento de coordenadas
