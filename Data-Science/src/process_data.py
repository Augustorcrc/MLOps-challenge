import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder
import logging
from process_geo import main_process_geo


def procesar_fechas_duracion(df):
    """Procesa fechas y calcula duración de atención."""
    logging.info("Procesando fechas y calculando duración de atención...")
    
    # Corregir duracion_min considerando zonas horarias y cruces de medianoche
    # 1) Parseos
    df['inicio_atencion_utc'] = pd.to_datetime(df['inicio_atencion_utc'], utc=True, errors='coerce')

    # fin_atencion en Paraguay
    fin_local = (
        pd.to_datetime(df['fin_atencion'], errors='coerce')
          .dt.tz_localize('America/Asuncion', nonexistent='shift_forward', ambiguous='NaT')
    )

    # 2) Comparar en misma TZ y corregir si cruzó medianoche y lo cargaron con mismo día
    inicio_local = df['inicio_atencion_utc'].dt.tz_convert('America/Asuncion')
    fin_local = fin_local.where(fin_local >= inicio_local, fin_local + pd.Timedelta(days=1))

    # 3) Llevar fin a UTC y calcular duración
    df['fin_atencion_utc'] = fin_local.dt.tz_convert('UTC')
    df['duracion_min'] = (df['fin_atencion_utc'] - df['inicio_atencion_utc']).dt.total_seconds() / 60

    # Verificar duracion_min negativa
    duraciones_negativas = (df['duracion_min'] < 0).sum()
    logging.info(f"Duraciones negativas después de corrección: {duraciones_negativas}")

    # Imputar duracion_min negativa
    df['duracion_min'] = df['duracion_min'].apply(lambda x: abs(x) if pd.notna(x) else x)

    logging.info("Procesamiento de fechas completado")
    return df


def procesar_variables_categoricas(df):
    """Procesa variables categóricas. Imputa valores faltantes."""
    logging.info("Procesando variables categóricas...")

    # Imputar tipo_asistencia faltante
    df['tipo_asistencia'] = df['tipo_asistencia'].fillna('desconocido')
    
    return df


def agregar_por_cliente(df):
    """Agrega datos por cliente_id.
    Crea una columna de conteo de veces que aparece el cliente
    y agrega columnas de modo, promedio, máximo y primer valor.
    """
    logging.info("Agregando datos por cliente_id...")
    
    # Crear columna de conteo de veces que aparece el cliente
    df['veces_cliente'] = df.groupby('cliente_id')['cliente_id'].transform('count')
    
    # Columnas para agregar con modo (más frecuente)
    mode_columns = ['segmento_cliente', 'tipo_asistencia']
    
    # Columnas para agregar con promedio
    mean_columns = ['duracion_min']

    max_columns = ['puntos_de_loyalty']
    
    # Columnas para agregar con primer valor (mantener estructura)
    first_columns = [col for col in df.columns 
                    if col not in mode_columns + mean_columns + max_columns + ['cliente_id', 'veces_cliente']]
    
    # Crear diccionario de agregaciones
    agg_dict = {}
    
    # Agregar columnas de modo
    for col in mode_columns:
        if col in df.columns:
            agg_dict[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    
    # Agregar columnas de promedio
    for col in mean_columns:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # Agregar columnas de máximo
    for col in max_columns:
        if col in df.columns:
            agg_dict[col] = 'max'
    
    # Agregar columnas de primer valor
    for col in first_columns:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    
    # Agregar lógica especial para churn (True si aparece algún True, False si todas son False)
    if 'churn' in df.columns:
        agg_dict['churn'] = lambda x: True if x.any() else False
    
    # Aplicar agregación
    df_agregado = df.groupby('cliente_id').agg(agg_dict).reset_index()
    
    logging.info(f"Dataset agregado por cliente_id: {df_agregado.shape}")
    logging.info(f"Clientes únicos: {df_agregado['cliente_id'].nunique()}")
    
    return df_agregado


def aplicar_encoding(df):
    """Aplica encoding ordinal y one-hot a variables categóricas."""
    logging.info("Aplicando encoding...")
    
    # Definir orden de categorías para segmento_cliente
    COL = 'segmento_cliente'
    letter_order = ['A', 'B', 'C', 'D']

    def sort_key(s):
        """Ordena las categorías de segmento_cliente."""
        if pd.isna(s):
            return (len(letter_order), 10**9)
        m = re.fullmatch(r'([A-D])(\d+)', str(s))
        if not m:
            return (len(letter_order), 10**9)
        L, n = m.group(1), int(m.group(2))
        return (letter_order.index(L), n)

    cats = sorted(pd.Series(df[COL]).dropna().unique().tolist(), key=sort_key)

    # Encoder ordinal con manejo de desconocidos
    oe = OrdinalEncoder(
        categories=[cats],
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=int
    )

    # Fit + transform
    df['segmento_cliente_ord'] = oe.fit_transform(df[[COL]]).ravel()

    # Invertir para que MAYOR = más valioso (A1 = máximo)
    max_code = len(oe.categories_[0]) - 1
    valid = df['segmento_cliente_ord'] >= 0
    df['segmento_cliente_ord'] = np.where(valid, max_code - df['segmento_cliente_ord'], -1)

    logging.info(f"Encoder ordinal creado con {len(cats)} categorías")

    # Aplicar one-hot encoding
    logging.info("Aplicando one-hot encoding...")
    
    # Para tipo de asistencia
    dummies = pd.get_dummies(df['tipo_asistencia'], prefix='tipo', drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    # Para zona_designada
    dummies_zona = pd.get_dummies(df['zona_asignada'], prefix='zona', drop_first=False)
    df = pd.concat([df, dummies_zona], axis=1)

    logging.info(f"One-hot encoding completado. Nuevo shape: {df.shape}")
    
    return df


def procesar_datos(df, df_zonas):
    """Función principal que procesa y prepara los datos para el entrenamiento."""
    try:
        logging.info("Procesando coordenadas y asignando zonas...")
        df_resultado = main_process_geo(
            df, 'coordenadas_sucursal',
            df_zonas, 'zona', 'poligono',
            debug=False
        )

        # Limpiar columnas innecesarias
        df_resultado.drop(columns=['latitud', 'longitud'], inplace=True)
        logging.info(f"Dataset procesado: {df_resultado.shape}")

        # Procesar fechas y duración
        df_resultado = procesar_fechas_duracion(df_resultado)

        # Procesar variables categóricas
        df_resultado = procesar_variables_categoricas(df_resultado)

        # Agregar por cliente
        df_resultado = agregar_por_cliente(df_resultado)

        # Aplicar encoding
        df_resultado = aplicar_encoding(df_resultado)
        
        return df_resultado
        
    except Exception as e:
        logging.error(f"Error en el procesamiento de datos: {e}")
        raise


if __name__ == "__main__":
    
    df = pd.read_csv('../data/dataset_churn_challenge.csv')
    df_zonas = pd.read_csv('../data/dataset_churn_zona_challenge.csv')
    df_resultado = procesar_datos(df, df_zonas)
    print(df_resultado.head())

    