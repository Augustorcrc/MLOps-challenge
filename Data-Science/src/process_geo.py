import pandas as pd
import json
import ast
import os
import logging


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

def main_process_geo(df_puntos, col_coordenadas_puntos, 
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
    cache_file = 'data_cache/coordenadas_procesadas_cache.csv'
    
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
        os.makedirs('data_cache', exist_ok=True)
        
        # Guardar resultado procesado
        df_con_zonas.to_csv(cache_file, index=False)
        logging.info(f"Caché guardado en: {cache_file}")
        
    except Exception as e:
        logging.warning(f"No se pudo guardar caché: {e}")
    
    return df_con_zonas


if __name__ == "__main__":
    df_puntos = pd.read_csv('../data/dataset_churn_challenge.csv')
    df_zonas = pd.read_csv('../data/dataset_churn_zona_challenge.csv')
    df_puntos_expandido = main_process_geo(df_puntos, 'coordenadas_sucursal', df_zonas, 'zona', 'poligono', debug=True)
    print(df_puntos_expandido.head())
