#!/usr/bin/env python3
"""
Utilidades para el modelo de predicción de churn.
Este módulo contiene todas las funciones auxiliares para el proyecto.
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
        df = pd.read_csv('../data/dataset_churn_challenge.csv')
        df_zonas = pd.read_csv('../data/dataset_churn_zona_challenge.csv')
        logging.info(f"Dataset principal cargado: {df.shape}")
        logging.info(f"Dataset de zonas cargado: {df_zonas.shape}")
        return df, df_zonas
    except FileNotFoundError as e:
        logging.error(f"Error al cargar datasets: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado al cargar datos: {e}")
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
