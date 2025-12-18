"""
Auto-Retrain Module
Automatically retrains the model when data drift is detected.
New model is saved as challenger for A/B testing.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.train_model import train_model, prepare_features, load_data
from ml.ab_testing import create_challenger_from_retrain

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configuration
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', '0.3'))  # 30% drift triggers retrain
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MLFLOW_TRACKING_URI = 'http://localhost:5000'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_fresh_training_data() -> Optional[pd.DataFrame]:
    """
    Fetch fresh data from CryptoCompare API for retraining.
    """
    try:
        from ml.fetch_training_data import fetch_hourly_ohlcv, calculate_features, CRYPTO_SYMBOLS
        
        logger.info("Fetching fresh training data from CryptoCompare...")
        
        all_data = []
        
        for crypto_name, symbol in CRYPTO_SYMBOLS.items():
            try:
                df = fetch_hourly_ohlcv(symbol, days=90)
                if df is not None and len(df) > 0:
                    df['crypto_id'] = crypto_name
                    df['symbol'] = symbol
                    df = calculate_features(df)
                    all_data.append(df)
                    logger.info(f"  {symbol}: {len(df)} records")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")
        
        if not all_data:
            logger.error("No data fetched!")
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Save fresh data
        output_path = os.path.join(DATA_DIR, 'fresh_training_data.csv')
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} records to {output_path}")
        
        return combined
        
    except Exception as e:
        logger.error(f"Failed to fetch fresh data: {e}")
        return None


def retrain_model(data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Retrain model with fresh data.
    Saves as challenger model for A/B testing.
    """
    result = {
        'success': False,
        'timestamp': datetime.now().isoformat(),
        'model_path': None,
        'metrics': {}
    }
    
    try:
        # Fetch fresh data if not provided
        if data is None:
            data = fetch_fresh_training_data()
            if data is None:
                # Fall back to existing data
                data_path = os.path.join(DATA_DIR, 'combined_binance.csv')
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
                    logger.info(f"Using existing data: {len(data)} records")
                else:
                    result['error'] = "No training data available"
                    return result
        
        # Load feature columns from train_model
        from ml.train_model import FEATURE_COLUMNS_BINANCE
        
        # Prepare data
        logger.info("Preparing training data...")
        X, y, feature_names = prepare_features(data, FEATURE_COLUMNS_BINANCE)
        
        if len(X) < 100:
            result['error'] = f"Insufficient data: {len(X)} samples"
            return result
        
        logger.info(f"Training on {len(X)} samples with {len(feature_names)} features")
        
        # Train model
        model, scaler, metrics, params = train_model(X, y, feature_names)
        
        # Save as challenger
        challenger_path = create_challenger_from_retrain(model, scaler)
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                mlflow.set_experiment('crypto-price-prediction')
                
                with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.log_param('trigger', 'auto_retrain')
                    mlflow.log_param('model_role', 'challenger')
                    mlflow.sklearn.log_model(model, 'challenger_model')
                    mlflow.end_run(status='FINISHED')
                    
                logger.info("Logged retrain run to MLflow")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
        result['success'] = True
        result['model_path'] = challenger_path
        result['metrics'] = metrics
        result['samples'] = len(X)
        
        logger.info(f"Retrain complete! Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        return result


def check_and_retrain(drift_share: float) -> Dict:
    """
    Check if drift exceeds threshold and trigger retrain.
    Called by drift_detector.py.
    """
    result = {
        'drift_share': drift_share,
        'threshold': DRIFT_THRESHOLD,
        'retrain_triggered': False
    }
    
    if drift_share >= DRIFT_THRESHOLD:
        logger.warning(f"⚠️ Drift threshold exceeded: {drift_share:.1%} >= {DRIFT_THRESHOLD:.1%}")
        logger.info("🔄 Triggering automatic retrain...")
        
        retrain_result = retrain_model()
        result['retrain_triggered'] = True
        result['retrain_result'] = retrain_result
        
        if retrain_result['success']:
            logger.info("✅ Auto-retrain successful! Challenger model ready for A/B testing.")
        else:
            logger.error(f"❌ Auto-retrain failed: {retrain_result.get('error', 'Unknown')}")
    else:
        logger.info(f"✅ Drift within acceptable range: {drift_share:.1%} < {DRIFT_THRESHOLD:.1%}")
    
    return result


def main():
    """Manual retrain entry point."""
    print("\n" + "=" * 60)
    print("AUTO-RETRAIN MODULE")
    print("=" * 60)
    
    print(f"\nDrift threshold: {DRIFT_THRESHOLD:.0%}")
    print(f"This creates a CHALLENGER model for A/B testing.")
    
    # Check for --force flag
    force = '--force' in sys.argv
    
    if force:
        print("\n🔄 Force retrain requested...")
        result = retrain_model()
    else:
        print("\nTo force retrain without drift check:")
        print("  python ml/auto_retrain.py --force")
        
        # Run drift check
        print("\nRunning drift detection first...")
        try:
            from monitoring.drift_detector import load_reference_data, fetch_production_data, detect_drift
            
            ref_df = load_reference_data()
            prod_df = fetch_production_data(minutes=60)
            
            if prod_df is not None and len(prod_df) >= 10:
                drift_result = detect_drift(ref_df, prod_df)
                drift_share = drift_result.get('drift_share', 0)
                result = check_and_retrain(drift_share)
            else:
                print("Not enough production data for drift check.")
                print("Run with --force to retrain anyway.")
                return
        except Exception as e:
            print(f"Drift check failed: {e}")
            print("Run with --force to retrain anyway.")
            return
    
    # Print results
    if 'retrain_result' in result or force:
        retrain_result = result.get('retrain_result', result) if not force else result
        print("\n" + "=" * 60)
        print("RETRAIN RESULTS")
        print("=" * 60)
        
        if retrain_result.get('success'):
            print(f"\n✅ Challenger model created!")
            print(f"📁 Path: {retrain_result.get('model_path')}")
            print(f"📊 Accuracy: {retrain_result.get('metrics', {}).get('accuracy', 0):.4f}")
            print(f"📈 Samples: {retrain_result.get('samples', 0):,}")
            print("\n🔄 A/B testing will now use both champion and challenger models.")
        else:
            print(f"\n❌ Retrain failed: {retrain_result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
