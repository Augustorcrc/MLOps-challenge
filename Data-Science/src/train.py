import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, 
                            classification_report, recall_score, precision_score, f1_score)
import xgboost as xgb
import os


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

        # Crear directorio results si no existe
        os.makedirs('results', exist_ok=True)
        
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
        
        # Guardar feature importance
        feature_importance_df.to_csv('results/feature_importance.csv', index=False)
        
        return xgb_model, y_pred, y_prob
        
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


def main_train(df_resultado, output_dir='results'):
    """
    Función principal para entrenar el modelo de churn.
    
    Args:
        df_resultado: DataFrame procesado con los datos listos para entrenar
        output_dir: Directorio donde guardar los resultados
    
    Returns:
        tuple: (modelo_entrenado, dataframe_con_predicciones)
    """
    try:
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{output_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info("=== INICIANDO ENTRENAMIENTO DEL MODELO ===")
        logging.info(f"Dataset de entrada: {df_resultado.shape}")
        
        # Paso 1: Preparar datos para entrenamiento
        X_train, X_test, y_train, y_test, X, y = preparar_datos_entrenamiento(df_resultado)
        
        # Paso 2: Entrenar modelo
        modelo, y_pred, y_prob = entrenar_modelo_seleccionado(
            X_train, X_test, y_train, y_test, df_resultado
        )
        
        # Paso 3: Crear DataFrame final con predicciones
        df_final = crear_dataframe_predicciones(
            df_resultado, X_train, X_test, y_test, y_train, y_pred, y_prob
        )
        
        # Paso 4: Guardar resultados
        logging.info("Guardando resultados...")
        df_final.to_csv(f'{output_dir}/predicciones_finales.csv', index=False)
        
        # Guardar modelo (opcional)
        import joblib
        joblib.dump(modelo, f'{output_dir}/modelo_xgboost.pkl')
        
        logging.info("=== ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")
        logging.info(f"Archivos guardados en: {output_dir}/")
        logging.info(f"- predicciones_finales.csv: {df_final.shape}")
        logging.info(f"- modelo_xgboost.pkl: Modelo entrenado")
        logging.info(f"- metricas.csv: Métricas de evaluación")
        logging.info(f"- feature_importance.csv: Importancia de features")
        logging.info(f"- training.log: Log del entrenamiento")
        
        return modelo, df_final
        
    except Exception as e:
        logging.error(f"Error en el proceso de entrenamiento: {e}")
        raise


if __name__ == "__main__":
    # Ejemplo de uso - ajustar según tus archivos de entrada
    try:
        # Cargar datos procesados (ajustar la ruta según tu caso)
        print("Cargando datos procesados...")
        df_resultado = pd.read_csv('data/datos_procesados.csv')  # Ajustar ruta
        
        # Ejecutar entrenamiento
        modelo, df_con_predicciones = main_train(df_resultado)
        
        print("\n¡Entrenamiento completado exitosamente!")
        print(f"Modelo guardado y DataFrame final con {df_con_predicciones.shape[0]} registros")
        
    except FileNotFoundError:
        print("Error: No se encontró el archivo de datos procesados.")
        print("Por favor, ajusta la ruta del archivo en el script o proporciona el DataFrame como parámetro.")
        print("\nEjemplo de uso programático:")
        print("modelo, df_final = main(tu_dataframe_procesado)")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        logging.error(f"Error durante la ejecución: {e}")