#!/usr/bin/env python3
"""
Script principal para el modelo de predicción de churn.
Este archivo contiene la función main que orquesta todo el proceso
de entrenamiento del modelo.
"""

from utils import (
    setup_logging,
    cargar_datos,
    analizar_datos,
    guardar_resultados,
    predecir_churn,
)
from process_data import procesar_datos
from train import main_train

def main():
    """Función principal del script."""
    try:
        # Configurar logging
        global logger
        logger = setup_logging()
        
        logger.info("Iniciando proceso de entrenamiento de modelo de churn")
        
        # Cargar datos
        df, df_zonas = cargar_datos()
        
        # Procesar datos
        df_resultado = procesar_datos(df, df_zonas)
        
        # Analizar datos
        analizar_datos(df_resultado)

        # Predecir churn actual
        df_resultado['prediccion_antigua'] = df_resultado['tipo_asistencia'].apply(predecir_churn)
        
        # Entrenar modelo
        modelo, df_final = main_train(df_resultado)
        
        # Guardar resultados
        guardar_resultados(df_final)
        
        # Resumen final
        logger.info("Proceso de entrenamiento completado exitosamente")
        logger.info("Resumen final:")
        logger.info(f"- Dataset original: {df.shape}")
        logger.info(f"- Dataset procesado: {df_resultado.shape}")
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":
    main()
