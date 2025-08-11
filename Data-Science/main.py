#!/usr/bin/env python3
"""
Script principal para el modelo de predicción de churn.
Este archivo contiene la función main que orquesta todo el proceso
de entrenamiento del modelo.
"""

from utils import (
    setup_logging,
    cargar_datos,
    procesar_datos,
    analizar_datos,
    preparar_datos_entrenamiento,
    entrenar_modelo_seleccionado,
    crear_dataframe_predicciones,
    guardar_resultados,
    predecir_churn
)


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
        
        # Preparar datos para entrenamiento
        X_train, X_test, y_train, y_test, X_full, y_full = preparar_datos_entrenamiento(df_resultado)
        
        # Entrenar mejor modelo
        y_pred, y_prob = entrenar_modelo_seleccionado(X_train, X_test, y_train, y_test, df_resultado)
        
        # Crear DataFrame con predicciones
        df_full = crear_dataframe_predicciones(df_resultado, X_train, X_test, y_test, y_train, y_pred, y_prob)
        
        # Guardar resultados
        guardar_resultados(df_full)
        
        # Resumen final
        logger.info("Proceso de entrenamiento completado exitosamente")
        logger.info("Resumen final:")
        logger.info(f"- Dataset original: {df.shape}")
        logger.info(f"- Dataset procesado: {df_resultado.shape}")
        logger.info(f"- Features para entrenamiento: {X_full.shape}")
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":
    main()
