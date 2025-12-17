"""
Train Price Direction Prediction Model with MLflow Tracking
Random Forest classifier for predicting crypto price movements.
Integrates with MLflow for experiment tracking and model registry.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Run: pip install mlflow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = 'crypto-price-prediction'

# Feature columns for Binance data (expanded features)
FEATURE_COLUMNS_BINANCE = [
    'price_change_1h',
    'price_change_4h',
    'price_change_24h',
    'sma_ratio_5_15',
    'sma_ratio_5_24',
    'price_to_sma_5',
    'price_to_sma_50',
    'rsi',
    'macd',
    'macd_signal',
    'macd_histogram',
    'bb_position',
    'volatility_5',
    'volatility_24',
    'volume_ratio',
    'momentum_5',
    'momentum_12',
    'momentum_24',
    'roc_5',
    'roc_12',
    'atr'
]

# Basic feature columns (fallback for CoinGecko data)
FEATURE_COLUMNS_BASIC = [
    'price_change_1h',
    'price_change_4h',
    'price_change_24h',
    'sma_ratio_5_15',
    'sma_ratio_5_24',
    'price_to_sma_5',
    'rsi',
    'volatility_5',
    'volatility_24',
    'volume_ratio',
    'momentum_5',
    'momentum_12'
]


def load_data(use_binance: bool = True) -> Tuple[pd.DataFrame, list]:
    """
    Load training data. Prefers Binance data if available.
    Returns DataFrame and list of feature columns to use.
    """
    binance_path = os.path.join(DATA_DIR, 'combined_binance.csv')
    coingecko_path = os.path.join(DATA_DIR, 'combined_historical.csv')
    
    if use_binance and os.path.exists(binance_path):
        df = pd.read_csv(binance_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} records from Binance data")
        return df, FEATURE_COLUMNS_BINANCE
    elif os.path.exists(coingecko_path):
        df = pd.read_csv(coingecko_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} records from CoinGecko data (fallback)")
        return df, FEATURE_COLUMNS_BASIC
    else:
        raise FileNotFoundError(
            "No training data found. Run fetch_binance_data.py first."
        )


def prepare_features(df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target vector.
    """
    # Filter to only features that exist in the dataframe
    available_features = [f for f in feature_cols if f in df.columns]
    logger.info(f"Using {len(available_features)} features")
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=available_features + ['target'])
    logger.info(f"Records after dropping NaN: {len(df_clean)}")
    
    if len(df_clean) == 0:
        raise ValueError("No valid training samples after filtering")
    
    X = df_clean[available_features].values
    y = df_clean['target'].values.astype(int)
    
    return X, y, available_features


def train_model(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: list,
    model_type: str = 'random_forest'
) -> Tuple[object, StandardScaler, dict]:
    """
    Train classifier with MLflow tracking.
    """
    n_samples = len(X)
    
    # Determine test size based on data availability
    test_size = 0.2 if n_samples >= 100 else 0.15
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model hyperparameters
    if model_type == 'random_forest':
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        model = RandomForestClassifier(**params)
    else:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'random_state': 42
        }
        model = GradientBoostingClassifier(**params)
    
    logger.info(f"Training {model_type}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': len(feature_names)
    }
    
    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"\nModel: {model_type}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall: {metrics['recall_weighted']:.4f}")
    print(f"F1 Score: {metrics['f1_weighted']:.4f}")
    
    # Classification report
    unique_classes = sorted(set(y_train) | set(y_test))
    target_names = ['DOWN', 'NEUTRAL', 'UP']
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=[target_names[c+1] for c in unique_classes],
        zero_division=0
    ))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nTop 10 Feature Importance:")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return model, scaler, metrics, params


def log_to_mlflow(
    model, 
    scaler, 
    metrics: dict, 
    params: dict, 
    feature_names: list,
    model_type: str
):
    """
    Log experiment to MLflow.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping logging")
        return
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param('model_type', model_type)
            mlflow.log_param('n_features', len(feature_names))
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log feature names
            mlflow.log_text('\n'.join(feature_names), 'feature_names.txt')
            
            # Log model (without registry to avoid version compatibility issues)
            mlflow.sklearn.log_model(model, 'model')
            
            # Log scaler
            mlflow.sklearn.log_model(scaler, 'scaler')
            
            # End run as successful
            mlflow.end_run(status='FINISHED')
            
            logger.info(f"Logged to MLflow: {MLFLOW_TRACKING_URI}")
            
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")
        logger.info("Continuing without MLflow logging...")


def save_model(model, scaler, feature_names: list, metrics: dict):
    """
    Save model locally.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'price_predictor.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save config
    config = {
        'feature_columns': feature_names,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'target_mapping': {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'},
        'metrics': metrics
    }
    config_path = os.path.join(MODELS_DIR, 'model_config.pkl')
    joblib.dump(config, config_path)
    logger.info(f"Saved config to {config_path}")


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*60)
    print("CRYPTO PRICE DIRECTION PREDICTION")
    print("Model Training Pipeline with MLflow")
    print("="*60 + "\n")
    
    # Load data (prefer Binance)
    df, feature_cols = load_data(use_binance=True)
    
    # Prepare features
    X, y, available_features = prepare_features(df, feature_cols)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nTarget distribution:")
    for val, count in zip(unique, counts):
        label = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}[val]
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train model
    model, scaler, metrics, params = train_model(
        X, y, available_features, model_type='random_forest'
    )
    
    # Log to MLflow if available
    log_to_mlflow(model, scaler, metrics, params, available_features, 'random_forest')
    
    # Save locally
    save_model(model, scaler, available_features, metrics)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: models/price_predictor.pkl")
    if MLFLOW_AVAILABLE:
        print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
    print("="*60)


if __name__ == '__main__':
    main()
