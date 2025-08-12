#!/usr/bin/env python3
"""
Script de prueba para verificar la nueva lógica de agregación por cliente_id
en la función procesar_datos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.utils import (
    setup_logging
)
from src.process_data import procesar_datos
from src.process_geo import main_process_geo

def cargar_datos_test(logger):
    """Carga los datasets necesarios para el entrenamiento."""
    try:
        logger.info("Cargando datasets...")
        df = pd.read_csv('./data/dataset_churn_challenge.csv')
        df_zonas = pd.read_csv('./data/dataset_churn_zona_challenge.csv')
        logger.info(f"Dataset principal cargado: {df.shape}")
        logger.info(f"Dataset de zonas cargado: {df_zonas.shape}")
        return df, df_zonas
    except FileNotFoundError as e:
        logger.error(f"Error al cargar datasets: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al cargar datos: {e}")
        raise



def test_aggregation():
    """Prueba la nueva lógica de agregación por cliente_id"""
    
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando prueba de agregación por cliente_id...")
    
    try:
        # Cargar datos
        df, df_zonas = cargar_datos_test(logger)
        logger.info(f"Datos originales cargados: {df.shape}")
        logger.info(f"Clientes únicos originales: {df['cliente_id'].nunique()}")
        
        # Verificar duplicados antes del procesamiento
        duplicados_antes = df['cliente_id'].duplicated().sum()
        logger.info(f"Clientes duplicados antes del procesamiento: {duplicados_antes}")
        
        # Procesar datos (esto ahora incluye la agregación)
        df_resultado = procesar_datos(df, df_zonas)
        logger.info(f"Datos procesados y agregados: {df_resultado.shape}")
        logger.info(f"Clientes únicos después de agregación: {df_resultado['cliente_id'].nunique()}")
        
        # Verificar que no hay duplicados después del procesamiento
        duplicados_despues = df_resultado['cliente_id'].duplicated().sum()
        logger.info(f"Clientes duplicados después del procesamiento: {duplicados_despues}")
        
        # Verificar que la columna veces_cliente existe
        if 'veces_cliente' in df_resultado.columns:
            logger.info("✅ Columna 'veces_cliente' encontrada")
            
            # Verificar algunos valores
            min_veces = df_resultado['veces_cliente'].min()
            max_veces = df_resultado['veces_cliente'].max()
            logger.info(f"Rango de veces_cliente: {min_veces} a {max_veces}")
            
            # Verificar que coincide con el conteo original
            if df_resultado['veces_cliente'].sum() == len(df):
                logger.info("✅ Suma de veces_cliente coincide con el total de registros originales")
            else:
                logger.warning("❌ Suma de veces_cliente NO coincide con el total original")
        else:
            logger.error("❌ Columna 'veces_cliente' NO encontrada")
        
        # Verificar que las columnas clave existen
        expected_columns = ['segmento_cliente', 'tipo_asistencia', 'duracion_min']
        for col in expected_columns:
            if col in df_resultado.columns:
                logger.info(f"✅ Columna {col} encontrada")
            else:
                logger.warning(f"❌ Columna {col} NO encontrada")
        
        # Verificar que no hay valores nulos en cliente_id
        null_clientes = df_resultado['cliente_id'].isnull().sum()
        logger.info(f"Valores nulos en cliente_id: {null_clientes}")
        
        logger.info("Prueba de agregación completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en la prueba de agregación: {e}")
        raise


if __name__ == "__main__":
    test_aggregation()
