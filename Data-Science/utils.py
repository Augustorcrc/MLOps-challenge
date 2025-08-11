#!/usr/bin/env python3
"""
Utilidades para el modelo de predicción de churn.
Este módulo contiene todas las funciones auxiliares para el procesamiento de datos,
entrenamiento de modelos y análisis.
"""

import pandas as pd
import random
import json
import ast
import re
import numpy as np
import logging
import warnings
import os
from pathlib import Path

# Imports de sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, 
    balanced_accuracy_score, classification_report,
    recall_score, precision_score, f1_score
)

# Imports de visualización
import matplotlib
matplotlib.use('Agg')  # Backend no-interactivo para entornos sin GUI
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Configura el sistema de logging para el script."""
    # Crear directorio de logs si no existe
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'model_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def predecir_churn(tipo_asistencia):
    """
    Función simple para predecir churn basada en tipo de asistencia.
    
    Args:
        tipo_asistencia: Tipo de asistencia del cliente
        
    Returns:
        float: Probabilidad de churn (0.0 a 1.0)
    """
    tipo_asistencia = str(tipo_asistencia)
    if tipo_asistencia.lower() == 'problema':
        return 1.0  # Siempre devuelve 1 si es un problema
    elif tipo_asistencia.lower() == 'reclamo':
        return 0.8 if random.random() < 0.5 else 0  # 50% de probabilidad para reclamos
    else:
        return 0

def punto_en_poligono(punto_lat, punto_lon, poligono_coords):
    """
    Determina si un punto está dentro de un polígono usando el algoritmo ray casting
    
    Args:
        punto_lat: Latitud del punto a verificar
        punto_lon: Longitud del punto a verificar
        poligono_coords: Lista de tuplas (lon, lat) que definen el polígono
    
    Returns:
        bool: True si el punto está dentro del polígono, False si no
    """
    x, y = punto_lon, punto_lat  # Punto a verificar (x=longitud, y=latitud)
    n = len(poligono_coords)
    dentro = False
    
    # Asegurar que tenemos al menos 3 puntos para formar un polígono
    if n < 3:
        return False
    
    j = n - 1  # Último índice
    
    for i in range(n):
        xi, yi = poligono_coords[i][0], poligono_coords[i][1]  # poligono_coords[i] = (lon, lat)
        xj, yj = poligono_coords[j][0], poligono_coords[j][1]  # poligono_coords[j] = (lon, lat)
        
        # Ray casting: contar intersecciones con una línea horizontal desde el punto
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            dentro = not dentro
        
        j = i
    
    return dentro

def procesar_coordenadas_json(df, col_coordenadas, col_identificador=None):
    """
    Convierte un DataFrame con strings JSON de coordenadas a formato expandido
    manteniendo todas las columnas originales del DataFrame
    
    Args:
        df: DataFrame original
        col_coordenadas: Nombre de la columna que contiene las coordenadas como JSON string
        col_identificador: Nombre de la columna identificadora (opcional, no usado en esta versión)
    
    Returns:
        DataFrame: DataFrame con todas las columnas originales más 'latitud' y 'longitud'
    """
    
    # Crear una copia del DataFrame original
    df_resultado = df.copy()
    
    # Listas para almacenar latitudes y longitudes
    latitudes = []
    longitudes = []
    
    for index, row in df.iterrows():
        coord_string = row[col_coordenadas]
        
        try:
            # Convertir string JSON a diccionario
            coord_dict = json.loads(coord_string)
            
            # Extraer latitud y longitud
            latitudes.append(coord_dict['latitud'])
            longitudes.append(coord_dict['longitud'])
                
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Error procesando coordenadas en fila {index}: {e}")
            # En caso de error, agregar valores NaN
            latitudes.append(pd.NA)
            longitudes.append(pd.NA)
            continue
    
    # Agregar las nuevas columnas al DataFrame
    df_resultado['latitud'] = latitudes
    df_resultado['longitud'] = longitudes
    
    return df_resultado

def determinar_zona(df_puntos, col_lat, col_lon, df_zonas, col_zona, col_coordenadas_zona, debug=False):
    """
    Determina en qué zona se encuentra cada punto
    
    Args:
        df_puntos: DataFrame con los puntos (caso JSON procesado)
        col_lat, col_lon: Nombres de las columnas de latitud y longitud
        df_zonas: DataFrame con las zonas (caso tuplas)
        col_zona: Nombre de la columna que identifica la zona
        col_coordenadas_zona: Nombre de la columna con las coordenadas del polígono
        debug: Si True, imprime información de debug
    
    Returns:
        DataFrame: DataFrame original con columna adicional 'zona_asignada'
    """
    
    # Procesar las zonas para tener los polígonos listos
    zonas_poligonos = {}
    
    for _, row in df_zonas.iterrows():
        zona_nombre = row[col_zona]
        coord_string = row[col_coordenadas_zona]
        
        try:
            # Convertir string a lista de tuplas
            coordenadas = ast.literal_eval(coord_string)
            zonas_poligonos[zona_nombre] = coordenadas
            
            if debug:
                logging.debug(f"Zona {zona_nombre} tiene {len(coordenadas)} puntos")
                logging.debug(f"  Primer punto: {coordenadas[0]}")
                logging.debug(f"  Último punto: {coordenadas[-1]}")
                
        except (ValueError, SyntaxError) as e:
            logging.warning(f"Error procesando zona {zona_nombre}: {e}")
            continue
    
    # Crear copia del DataFrame de puntos
    df_resultado = df_puntos.copy()
    zonas_asignadas = []
    
    # Para cada punto, verificar en qué zona está
    for idx, row in df_puntos.iterrows():
        punto_lat = row[col_lat]
        punto_lon = row[col_lon]
        zona_encontrada = None
        
        if debug:
            logging.debug(f"\nVerificando punto {idx}: ({punto_lat}, {punto_lon})")
        
        # Verificar contra cada zona
        for zona_nombre, poligono_coords in zonas_poligonos.items():
            esta_dentro = punto_en_poligono(punto_lat, punto_lon, poligono_coords)
            
            if debug:
                logging.debug(f"  ¿Está en {zona_nombre}?: {esta_dentro}")
            
            if esta_dentro:
                zona_encontrada = zona_nombre
                break  # Asignar a la primera zona donde se encuentre
        
        if debug:
            logging.debug(f"  Zona asignada: {zona_encontrada}")
            
        zonas_asignadas.append(zona_encontrada)
    
    df_resultado['zona_asignada'] = zonas_asignadas
    return df_resultado

def procesar_coordenadas_completo(df_puntos, col_coordenadas_puntos, 
                                  df_zonas, col_zona, col_coordenadas_zonas, debug=False):
    """
    Función completa que procesa puntos JSON y los asigna a zonas
    Implementa caché para evitar reprocesamiento costoso
    
    Args:
        df_puntos: DataFrame con coordenadas JSON
        col_coordenadas_puntos: Columna con las coordenadas JSON
        df_zonas: DataFrame con polígonos de zonas
        col_zona: Columna con nombres de zonas
        col_coordenadas_zonas: Columna con coordenadas de polígonos
        debug: Si True, activa modo debug
    
    Returns:
        DataFrame: DataFrame con puntos expandidos y zonas asignadas
    """
    
    # Verificar si existe caché
    cache_file = 'data/coordenadas_procesadas_cache.csv'
    
    try:
        # Intentar cargar desde caché
        if os.path.exists(cache_file):
            logging.info("Cargando coordenadas procesadas desde caché...")
            df_cache = pd.read_csv(cache_file)
            
            # Verificar que el caché tenga el mismo número de registros
            if len(df_cache) == len(df_puntos):
                logging.info("Caché válido encontrado. Usando datos en caché.")
                return df_cache
            else:
                logging.warning(f"Caché inválido (registros: {len(df_cache)} vs {len(df_puntos)}). Reprocesando...")
        else:
            logging.info("No se encontró caché. Procesando coordenadas...")
            
    except Exception as e:
        logging.warning(f"Error al cargar caché: {e}. Reprocesando coordenadas...")
    
    # Si no hay caché válido, procesar normalmente
    logging.info("Procesando coordenadas y asignando zonas...")
    
    # Paso 1: Expandir coordenadas JSON
    df_puntos_expandido = procesar_coordenadas_json(df_puntos, col_coordenadas_puntos)
    
    # Paso 2: Asignar zonas
    df_con_zonas = determinar_zona(
        df_puntos_expandido, 'latitud', 'longitud',
        df_zonas, col_zona, col_coordenadas_zonas, debug=debug
    )
    
    # Guardar en caché para uso futuro
    try:
        # Crear directorio data si no existe
        os.makedirs('data', exist_ok=True)
        
        # Guardar resultado procesado
        df_con_zonas.to_csv(cache_file, index=False)
        logging.info(f"Caché guardado en: {cache_file}")
        
    except Exception as e:
        logging.warning(f"No se pudo guardar caché: {e}")
    
    return df_con_zonas

def outlier_summary_iqr(df, factor=1.5):
    """
    Resume la cantidad de outliers por columna numérica usando el método IQR.
    
    Args:
        df: DataFrame a analizar
        factor: Factor para determinar outliers (1.5 para comunes, 3.0 para extremos)
    
    Returns:
        tuple: (resumen, máscara de outliers)
    """
    num = df.select_dtypes(include='number')
    if num.empty:
        raise ValueError("No hay columnas numéricas.")

    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    mask = num.lt(lower) | num.gt(upper)   # True donde hay outlier
    summary = pd.DataFrame({
        'q1': q1, 'q3': q3, 'iqr': iqr,
        'upper': upper,
        'n_outliers': mask.sum(),
        '%_outliers': (mask.sum() / len(num) * 100).round(2)
    }).sort_values('%_outliers', ascending=False)

    return summary, mask

def cargar_datos():
    """Carga los datasets necesarios para el entrenamiento."""
    try:
        logging.info("Cargando datasets...")
        df = pd.read_csv('data/dataset_churn_challenge.csv')
        df_zonas = pd.read_csv('data/dataset_churn_zona_challenge.csv')
        logging.info(f"Dataset principal cargado: {df.shape}")
        logging.info(f"Dataset de zonas cargado: {df_zonas.shape}")
        return df, df_zonas
    except FileNotFoundError as e:
        logging.error(f"Error al cargar datasets: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado al cargar datos: {e}")
        raise

def procesar_datos(df, df_zonas):
    """Procesa y prepara los datos para el entrenamiento."""
    try:
        logging.info("Procesando coordenadas y asignando zonas...")
        df_resultado = procesar_coordenadas_completo(
            df, 'coordenadas_sucursal',
            df_zonas, 'zona', 'poligono',
            debug=False
        )

        df_resultado.drop(columns=['latitud', 'longitud'], inplace=True)
        logging.info(f"Dataset procesado: {df_resultado.shape}")

        # Procesar fechas y calcular duración de atención
        logging.info("Procesando fechas y calculando duración de atención...")
        
        # Corregir duracion_min considerando zonas horarias y cruces de medianoche
        # 1) Parseos
        df_resultado['inicio_atencion_utc'] = pd.to_datetime(df_resultado['inicio_atencion_utc'], utc=True, errors='coerce')

        # fin_atencion en Paraguay
        fin_local = (
            pd.to_datetime(df_resultado['fin_atencion'], errors='coerce')
              .dt.tz_localize('America/Asuncion', nonexistent='shift_forward', ambiguous='NaT')
        )

        # 2) Comparar en misma TZ y corregir si cruzó medianoche y lo cargaron con mismo día
        inicio_local = df_resultado['inicio_atencion_utc'].dt.tz_convert('America/Asuncion')
        fin_local = fin_local.where(fin_local >= inicio_local, fin_local + pd.Timedelta(days=1))

        # 3) Llevar fin a UTC y calcular duración
        df_resultado['fin_atencion_utc'] = fin_local.dt.tz_convert('UTC')
        df_resultado['duracion_min'] = (df_resultado['fin_atencion_utc'] - df_resultado['inicio_atencion_utc']).dt.total_seconds() / 60

        # Verificar duracion_min negativa
        duraciones_negativas = (df_resultado['duracion_min'] < 0).sum()
        logging.info(f"Duraciones negativas después de corrección: {duraciones_negativas}")

        # Imputar duracion_min negativa
        df_resultado['duracion_min'] = df_resultado['duracion_min'].apply(lambda x: abs(x) if pd.notna(x) else x)

        logging.info("Procesamiento de fechas completado")

        # Procesar variables categóricas
        logging.info("Procesando variables categóricas...")

         # Imputar tipo_asistencia faltante
        df_resultado['tipo_asistencia'] = df_resultado['tipo_asistencia'].fillna('desconocido')
        

        
  # Agregar paso de agregación por cliente_id
        logging.info("Agregando datos por cliente_id...")
        
        # Crear columna de conteo de veces que aparece el cliente
        df_resultado['veces_cliente'] = df_resultado.groupby('cliente_id')['cliente_id'].transform('count')
        

        
        # Columnas para agregar con modo (más frecuente)
        mode_columns = ['segmento_cliente', 'tipo_asistencia']
        
        # Columnas para agregar con promedio
        mean_columns = [ 'duracion_min']

        max_columns = ['puntos_de_loyalty']
        
        # Columnas para agregar con primer valor (mantener estructura)
        first_columns = [col for col in df_resultado.columns 
                        if col not in mode_columns + mean_columns + ['cliente_id', 'veces_cliente']]
        
        # Crear diccionario de agregaciones
        agg_dict = {}
        
        # Agregar columnas de modo
        for col in mode_columns:
            if col in df_resultado.columns:
                agg_dict[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        
        # Agregar columnas de promedio
        for col in mean_columns:
            if col in df_resultado.columns:
                agg_dict[col] = 'mean'

        # Agregar columnas de máximo
        for col in max_columns:
            if col in df_resultado.columns:
                agg_dict[col] = 'max'
        
        # Agregar columnas de primer valor
        for col in first_columns:
            if col in df_resultado.columns:
                agg_dict[col] = 'first'
        
        # Agregar columna de conteo
        agg_dict['veces_cliente'] = 'first'
        
        # Agregar lógica especial para churn (True si aparece algún True, False si todas son False)
        if 'churn' in df_resultado.columns:
            agg_dict['churn'] = lambda x: True if x.any() else False
        
        # Aplicar agregación
        df_resultado = df_resultado.groupby('cliente_id').agg(agg_dict).reset_index()
        
        logging.info(f"Dataset agregado por cliente_id: {df_resultado.shape}")
        logging.info(f"Clientes únicos: {df_resultado['cliente_id'].nunique()}")

        
        # Definir orden de categorías para segmento_cliente
        COL = 'segmento_cliente'
        letter_order = ['A', 'B', 'C', 'D']

        def sort_key(s):
            if pd.isna(s):
                return (len(letter_order), 10**9)
            m = re.fullmatch(r'([A-D])(\d+)', str(s))
            if not m:
                return (len(letter_order), 10**9)
            L, n = m.group(1), int(m.group(2))
            return (letter_order.index(L), n)

        cats = sorted(pd.Series(df_resultado[COL]).dropna().unique().tolist(), key=sort_key)

        # Encoder ordinal con manejo de desconocidos
        oe = OrdinalEncoder(
            categories=[cats],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=int
        )

        # Fit + transform
        df_resultado['segmento_cliente_ord'] = oe.fit_transform(df_resultado[[COL]]).ravel()

        # Invertir para que MAYOR = más valioso (A1 = máximo)
        max_code = len(oe.categories_[0]) - 1
        valid = df_resultado['segmento_cliente_ord'] >= 0
        df_resultado['segmento_cliente_ord'] = np.where(valid, max_code - df_resultado['segmento_cliente_ord'], -1)

        logging.info(f"Encoder ordinal creado con {len(cats)} categorías")

        # Aplicar one-hot encoding
        logging.info("Aplicando one-hot encoding...")
        
        # Para tipo de asistencia
        dummies = pd.get_dummies(df_resultado['tipo_asistencia'], prefix='tipo', drop_first=False)
        df_resultado = pd.concat([df_resultado, dummies], axis=1)

        # Para zona_designada
        dummies_zona = pd.get_dummies(df_resultado['zona_asignada'], prefix='zona', drop_first=False)
        df_resultado = pd.concat([df_resultado, dummies_zona], axis=1)

        logging.info(f"One-hot encoding completado. Nuevo shape: {df_resultado.shape}")
        
      
        
        return df_resultado
        
    except Exception as e:
        logging.error(f"Error en el procesamiento de datos: {e}")
        raise

def analizar_datos(df_resultado):
    """Realiza análisis exploratorio de los datos."""
    try:
        logging.info("Analizando distribución del target...")
        churn_dist = df_resultado.churn.value_counts(normalize=True)
        logging.info(f"Distribución del target (churn): {churn_dist.to_dict()}")

        logging.info("Información del dataset:")
        logging.info(f"Shape: {df_resultado.shape}")
        logging.info(f"Columnas: {list(df_resultado.columns)}")
        logging.info(f"Tipos de datos: {df_resultado.dtypes.value_counts().to_dict()}")

        # Analizar outliers
        logging.info("Analizando outliers...")
        resumen, mask = outlier_summary_iqr(df_resultado, factor=3.0)
        filas_con_outlier = df_resultado[mask.any(axis=1)]
        logging.info(f"Filas con outliers: {len(filas_con_outlier)}")
        logging.info("Resumen de outliers:")
        logging.info(resumen)

        # Generar matriz de correlación
        logging.info("Generando matriz de correlación...")
        df_corr = df_resultado.select_dtypes(include=['number', 'bool']).copy()
        bool_cols = df_corr.select_dtypes(include='bool').columns
        df_corr[bool_cols] = df_corr[bool_cols].astype(int)

        plt.figure(figsize=(12, 8))
        corr = df_corr.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title('Matriz de Correlación')
        
        # Crear directorio de resultados si no existe
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        plt.savefig(results_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar figura para liberar memoria
        logging.info("Matriz de correlación guardada como 'results/correlation_matrix.png'")

        # Analizar correlaciones con target
        TARGET = 'churn'
        TOP = 25

        logging.info(f"Analizando correlaciones con target: {TARGET}")
        X = df_resultado.select_dtypes(include=['number', 'bool']).copy()
        bool_cols = X.select_dtypes(include='bool').columns
        X[bool_cols] = X[bool_cols].astype(int)
        X = X.loc[:, X.nunique(dropna=False) > 1]

        if TARGET not in X.columns:
            raise ValueError(f"'{TARGET}' no está en las columnas numéricas/bool.")

        corr_s = X.corr(numeric_only=True, method='pearson')[TARGET].drop(TARGET).dropna()
        corr_top = corr_s.reindex(corr_s.abs().sort_values(ascending=False).index).head(TOP)

        tabla = (corr_top.rename('corr')
                 .to_frame()
                 .assign(abs=lambda d: d['corr'].abs()))

        logging.info('Top correlaciones con el target:')
        logging.info(tabla)
        
        return X
        
    except Exception as e:
        logging.error(f"Error en el análisis de datos: {e}")
        raise

def preparar_datos_entrenamiento(df_resultado):
    """Prepara los datos para el entrenamiento del modelo."""
    try:
        logging.info("Preparando datos para entrenamiento...")
        
        TARGET = 'churn'
        
        # Tomar numéricas + booleanas
        df_numerico = df_resultado.select_dtypes(include=['number', 'bool']).copy()

        # Castear booleanas a int
        bool_cols = df_numerico.select_dtypes(include='bool').columns
        df_numerico[bool_cols] = df_numerico[bool_cols].astype(int)

        # Verificar que no hay columnas con valores infinitos o NaN
        df_numerico = df_numerico.replace([np.inf, -np.inf], np.nan)
        df_numerico = df_numerico.dropna()

        logging.info(f"Dataset numérico final: {df_numerico.shape}")
        logging.info(f"Columnas disponibles para entrenamiento: {list(df_numerico.columns)}")

        # Verificar que no hay duplicados en cliente_id
        if df_numerico['cliente_id'].duplicated().any():
            logging.warning("Se encontraron cliente_id duplicados. Manteniendo solo el último.")
            df_numerico = df_numerico.drop_duplicates(subset=['cliente_id'], keep='last')
        else:
            logging.info("No se encontraron cliente_id duplicados.")

        X = df_numerico.drop(columns=[TARGET])
        y = df_numerico[TARGET]

        logging.info(f"Features (X): {X.shape}")
        logging.info(f"Target (y): {y.shape}")
        logging.info(f"Distribución del target: {y.value_counts(normalize=True).to_dict()}")

        # Split estratificado para mantener proporción de clases
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logging.info(f"Distribución train: {y_train.value_counts(normalize=True).to_dict()}")
        logging.info(f"Distribución test: {y_test.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_test, y_train, y_test, X, y
        
    except Exception as e:
        logging.error(f"Error preparando datos de entrenamiento: {e}")
        raise

def entrenar_modelo_seleccionado(X_train, X_test, y_train, y_test, df_resultado):
    """Entrena el mejor modelo basado en los resultados de evaluación usando XGBoost."""
    try:
        logging.info("Implementando XGBoost Classifier optimizado...")

        import xgboost as xgb
        import numpy as np
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report

        # Determinar si es problema binario o multiclase
        n_classes = len(np.unique(y_train))
        

        # Parámetros optimizados de XGBoost
        xgb_params = {    
            'random_state': 42,
            'verbosity': 0               # Silenciar warnings
        }
     
        # Manejo del desbalance de clases 
        if n_classes == 2:
            # Para problemas binarios, usar scale_pos_weight
            class_counts = np.bincount(y_train)
            scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
            xgb_params['scale_pos_weight'] = scale_pos_weight
            logging.info(f"Balanceo de clases: {scale_pos_weight}")
        else:
            # Para multiclase, XGBoost maneja automáticamente el desbalance
            pass

        logging.info("Entrenando XGBoost Classifier...")
        logging.info(f"Columnas de X_train: {X_train.columns}")
        
        # Crear y entrenar el modelo
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)

        # Predicciones
        y_pred = xgb_model.predict(X_test)

        y_pred_antigua = (X_test.reset_index()[['cliente_id']]
                          .merge(df_resultado.drop_duplicates('cliente_id', keep='last'),
                          on='cliente_id', how='left')['prediccion_antigua']
                          .apply(lambda x: 0 if x == 0.0 else 1)
                          .rename('y_pred_antigua'))


        # Metricas
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy_antigua = accuracy_score(y_test, y_pred_antigua)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Balanced Acc: {balanced_acc:.4f}")
        logging.info(f"Accuracy Antigua: {accuracy_antigua:.4f}")

        # Inicializar y_prob
        y_prob = None
        
        # Si es binario y hay predict_proba, medir AUC
        if n_classes == 2 and hasattr(xgb_model, "predict_proba"):
            y_prob = xgb_model.predict_proba(X_test)[:, 1]
            logging.info(f"ROC AUC: {roc_auc:.4f}")
        elif n_classes > 2 and hasattr(xgb_model, "predict_proba"):
            # Para multiclase, obtener todas las probabilidades
            y_prob = xgb_model.predict_proba(X_test)

        logging.info("Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # feature importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
             }).sort_values('importance', ascending=False)

        logging.info("Feature importance (ordenado):")
        for _, row in feature_importance_df.iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")

        # Guardar metricas en csv
        metricas = pd.DataFrame({
            'accuracy': [accuracy],
            'balanced_acc': [balanced_acc],
            'roc_auc': [roc_auc],
            'recall': [recall],
            'precision': [precision],
            'f1': [f1],
            'accuracy_antigua': [accuracy_antigua]
        })
        metricas.to_csv('results/metricas.csv', index=False)
        
        return y_pred, y_prob
        
    except Exception as e:
        logging.error(f"Error entrenando el mejor modelo: {e}")
        raise

def crear_dataframe_predicciones(df_resultado, X_train, X_test, y_test, y_train, y_pred, y_prob):
    """
    Crea un DataFrame con el df original más las predicciones del modelo
    usando un enfoque simple de merge por cliente_id.
    
    Args:
        df_resultado: DataFrame original procesado
        X_train: Features de entrenamiento
        X_test: Features de test
        y_test: Target de test
        y_train: Target de entrenamiento
        y_pred: Predicciones del modelo
        y_prob: Probabilidades del modelo
        
    Returns:
        DataFrame: DataFrame completo con predicciones y flag de train/test
    """
    try:
        logging.info("Creando DataFrame con predicciones usando merge por cliente_id...")
        
        # Paso 1: Pegar y_test e y_pred a X_test
        X_test_with_predictions = X_test.copy()
        X_test_with_predictions['y_test'] = y_test
        X_test_with_predictions['y_pred'] = y_pred
        X_test_with_predictions['y_prob'] = y_prob
        X_test_with_predictions['dataset_split'] = 'test'
        
        # Paso 2: Pegar y_train a X_train
        X_train_with_targets = X_train.copy()
        X_train_with_targets['y_train'] = y_train
        X_train_with_targets['dataset_split'] = 'train'
        
        # Paso 3: Concatenar ambos DataFrames
        df_combined = pd.concat([X_train_with_targets, X_test_with_predictions], axis=0)
        
        # Verificar que no hay duplicados en cliente_id
        if df_combined['cliente_id'].duplicated().any():
            logging.warning("Se encontraron cliente_id duplicados. Manteniendo solo el último.")
            df_combined = df_combined.drop_duplicates(subset=['cliente_id'], keep='last')
        
        # Paso 4: Merge por cliente_id con df_resultado
        df_final = df_resultado.merge(
            df_combined[['cliente_id', 'y_test', 'y_pred', 'y_prob', 'y_train', 'dataset_split']], 
            on='cliente_id', 
            how='left'
        )
        
        # Limpiar columnas y crear columnas finales
        df_final['target_real'] = df_final['churn']
        df_final['prediccion_churn'] = df_final['y_pred']
        df_final['probabilidad_churn'] = df_final['y_prob']
        
       
        # Limpiar columnas temporales
        df_final = df_final.drop(['y_test', 'y_pred', 'y_prob', 'y_train'], axis=1)
        
        logging.info(f"DataFrame de predicciones creado con shape: {df_final.shape}")
        logging.info(f"Registros de train: {(df_final['dataset_split'] == 'train').sum()}")
        logging.info(f"Registros de test: {(df_final['dataset_split'] == 'test').sum()}")
        
        return df_final
        
    except Exception as e:
        logging.error(f"Error creando DataFrame de predicciones: {e}")
        raise

def guardar_resultados(df_full):
    """Guarda los resultados del entrenamiento."""
    try:
        # Crear directorio de resultados si no existe
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Guardar DataFrame con predicciones
        df_full.to_csv(results_dir / 'predictions.csv', index=False)
        logging.info("Predicciones guardadas en 'results/predictions.csv'")
        
    except Exception as e:
        logging.error(f"Error guardando resultados: {e}")
        raise
