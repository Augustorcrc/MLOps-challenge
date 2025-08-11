#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad de caché
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import setup_logging, procesar_coordenadas_completo



def cargar_datos_test(logger):
    """Carga los datasets necesarios para el entrenamiento."""
    try:
        logger.info("Cargando datasets...")
        df = pd.read_csv('../data/dataset_churn_challenge.csv')
        df_zonas = pd.read_csv('../data/dataset_churn_zona_challenge.csv')
        logger.info(f"Dataset principal cargado: {df.shape}")
        logger.info(f"Dataset de zonas cargado: {df_zonas.shape}")
        return df, df_zonas
    except FileNotFoundError as e:
        logger.error(f"Error al cargar datasets: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al cargar datos: {e}")
        raise


def test_caching():
    """Prueba la funcionalidad de caché"""
    
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando prueba de caché...")
    
    try:
        # Cargar datos
        df, df_zonas = cargar_datos_test(logger)
        logger.info(f"Datos cargados: {df.shape}")
        
        # Primera ejecución - debería procesar y guardar caché
        logger.info("=== Primera ejecución ===")
        df_resultado1 = procesar_coordenadas_completo(
            df, 'coordenadas_sucursal',
            df_zonas, 'zona', 'poligono',
            debug=False
        )
        logger.info(f"Primera ejecución completada: {df_resultado1.shape}")
        
        # Verificar si se creó el archivo de caché
        cache_file = 'data/coordenadas_procesadas_cache.csv'
        if os.path.exists(cache_file):
            logger.info(f"✅ Archivo de caché creado: {cache_file}")
            cache_size = os.path.getsize(cache_file)
            logger.info(f"Tamaño del caché: {cache_size} bytes")
        else:
            logger.warning("❌ No se creó el archivo de caché")
        
        # Segunda ejecución - debería cargar desde caché
        logger.info("=== Segunda ejecución ===")
        df_resultado2 = procesar_coordenadas_completo(
            df, 'coordenadas_sucursal',
            df_zonas, 'zona', 'poligono',
            debug=False
        )
        logger.info(f"Segunda ejecución completada: {df_resultado2.shape}")
        
        # Verificar que ambos resultados sean iguales
        if df_resultado1.equals(df_resultado2):
            logger.info("✅ Los resultados son idénticos - caché funcionando correctamente")
        else:
            logger.warning("❌ Los resultados son diferentes - problema con el caché")
        
        logger.info("Prueba de caché completada")
        
    except Exception as e:
        logger.error(f"Error en la prueba: {e}")
        raise


if __name__ == "__main__":
    test_caching()
