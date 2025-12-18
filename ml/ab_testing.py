"""
A/B Testing Framework for Model Comparison
Compares champion (production) vs challenger (new) models with configurable traffic split.
"""

import os
import random
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import joblib

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, 'price_predictor.pkl')
CHALLENGER_MODEL_PATH = os.path.join(MODELS_DIR, 'challenger_model.pkl')
CHAMPION_SCALER_PATH = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
CHALLENGER_SCALER_PATH = os.path.join(MODELS_DIR, 'challenger_scaler.pkl')
CONFIG_PATH = os.path.join(MODELS_DIR, 'model_config.pkl')

# Traffic split
# 0.8 means 80% champion, 20% challenger
CHAMPION_TRAFFIC_RATIO = float(os.getenv('AB_CHAMPION_RATIO', '0.8'))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestPredictor:
    """
    A/B Testing Predictor that routes traffic between champion and challenger models.
    """
    
    def __init__(self, champion_ratio: float = CHAMPION_TRAFFIC_RATIO):
        self.champion_ratio = champion_ratio
        self.champion_model = None
        self.challenger_model = None
        self.champion_scaler = None
        self.challenger_scaler = None
        self.config = None
        
        # Metrics tracking
        self.metrics = {
            'champion': {'predictions': 0, 'correct': 0},
            'challenger': {'predictions': 0, 'correct': 0}
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load champion and challenger models."""
        # Load champion model (required)
        if os.path.exists(CHAMPION_MODEL_PATH):
            self.champion_model = joblib.load(CHAMPION_MODEL_PATH)
            logger.info(f"Loaded champion model from {CHAMPION_MODEL_PATH}")
        else:
            logger.warning("Champion model not found!")
        
        if os.path.exists(CHAMPION_SCALER_PATH):
            self.champion_scaler = joblib.load(CHAMPION_SCALER_PATH)
        
        # Load challenger model (optional)
        if os.path.exists(CHALLENGER_MODEL_PATH):
            self.challenger_model = joblib.load(CHALLENGER_MODEL_PATH)
            self.challenger_scaler = joblib.load(CHALLENGER_SCALER_PATH) if os.path.exists(CHALLENGER_SCALER_PATH) else self.champion_scaler
            logger.info(f"Loaded challenger model from {CHALLENGER_MODEL_PATH}")
        else:
            logger.info("No challenger model found. Running champion only.")
        
        # Load config
        if os.path.exists(CONFIG_PATH):
            self.config = joblib.load(CONFIG_PATH)
    
    def _select_model(self) -> str:
        """Select model based on traffic split."""
        if self.challenger_model is None:
            return 'champion'
        
        return 'champion' if random.random() < self.champion_ratio else 'challenger'
    
    def predict(self, features: Dict) -> Optional[Dict]:
        """
        Make prediction using A/B testing logic.
        Returns prediction with model version info.
        """
        if self.champion_model is None:
            return None
        
        if self.config is None:
            return None
        
        try:
            # Prepare features
            feature_cols = self.config['feature_columns']
            feature_values = []
            
            for col in feature_cols:
                val = features.get(col)
                if val is None:
                    return None
                feature_values.append(val)
            
            X = np.array([feature_values])
            
            # Select model variant
            variant = self._select_model()
            
            # Get model and scaler for variant
            if variant == 'champion':
                model = self.champion_model
                scaler = self.champion_scaler
            else:
                model = self.challenger_model
                scaler = self.challenger_scaler
            
            # Scale and predict
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Map to label
            label_map = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}
            
            # Update metrics
            self.metrics[variant]['predictions'] += 1
            
            result = {
                'prediction': label_map.get(prediction, 'NEUTRAL'),
                'prediction_raw': int(prediction),
                'probability': float(max(probabilities)),
                'probabilities': {
                    'DOWN': float(probabilities[0]) if len(probabilities) > 0 else 0,
                    'NEUTRAL': float(probabilities[1]) if len(probabilities) > 1 else 0,
                    'UP': float(probabilities[2]) if len(probabilities) > 2 else 0
                },
                'model_variant': variant,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                self._log_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"A/B prediction error: {e}")
            return None
    
    def _log_prediction(self, result: Dict):
        """Log prediction to MLflow for tracking."""
        try:
            mlflow.log_metric(f"ab_{result['model_variant']}_predictions", 
                            self.metrics[result['model_variant']]['predictions'])
        except Exception:
            pass  # Silently fail if MLflow not configured
    
    def get_metrics(self) -> Dict:
        """Get A/B testing metrics."""
        total_champion = self.metrics['champion']['predictions']
        total_challenger = self.metrics['challenger']['predictions']
        total = total_champion + total_challenger
        
        return {
            'champion_predictions': total_champion,
            'challenger_predictions': total_challenger,
            'champion_ratio_actual': total_champion / total if total > 0 else 0,
            'challenger_ratio_actual': total_challenger / total if total > 0 else 0,
            'total_predictions': total
        }
    
    def promote_challenger(self) -> bool:
        """
        Promote challenger model to champion.
        Call this when challenger outperforms champion.
        """
        if not os.path.exists(CHALLENGER_MODEL_PATH):
            logger.warning("No challenger model to promote")
            return False
        
        try:
            # Backup current champion
            backup_path = CHAMPION_MODEL_PATH.replace('.pkl', '_backup.pkl')
            if os.path.exists(CHAMPION_MODEL_PATH):
                import shutil
                shutil.copy(CHAMPION_MODEL_PATH, backup_path)
                logger.info(f"Backed up champion to {backup_path}")
            
            # Promote challenger
            import shutil
            shutil.copy(CHALLENGER_MODEL_PATH, CHAMPION_MODEL_PATH)
            if os.path.exists(CHALLENGER_SCALER_PATH):
                shutil.copy(CHALLENGER_SCALER_PATH, CHAMPION_SCALER_PATH)
            
            # Reload models
            self._load_models()
            
            logger.info("Challenger promoted to champion!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote challenger: {e}")
            return False


def create_challenger_from_retrain(model, scaler):
    """
    Save a newly trained model as the challenger for A/B testing.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(model, CHALLENGER_MODEL_PATH)
    joblib.dump(scaler, CHALLENGER_SCALER_PATH)
    
    logger.info(f"Saved challenger model to {CHALLENGER_MODEL_PATH}")
    return CHALLENGER_MODEL_PATH


# Singleton instance for use in stream processor
_ab_predictor = None

def get_ab_predictor() -> ABTestPredictor:
    """Get or create A/B testing predictor instance."""
    global _ab_predictor
    if _ab_predictor is None:
        _ab_predictor = ABTestPredictor()
    return _ab_predictor


if __name__ == '__main__':
    # Status Check - shows current A/B testing configuration
    print("=" * 60)
    print("A/B TESTING STATUS CHECK")
    print("=" * 60)
    
    predictor = get_ab_predictor()
    
    print(f"\nChampion traffic ratio: {predictor.champion_ratio:.0%}")
    print(f"Challenger traffic ratio: {1 - predictor.champion_ratio:.0%}")
    
    if predictor.champion_model:
        print(f"\n✅ Champion model loaded: {type(predictor.champion_model).__name__}")
    else:
        print("\n❌ No champion model found!")
    
    if predictor.challenger_model:
        print(f"✅ Challenger model loaded - A/B testing ACTIVE")
    else:
        print("⚠️ No challenger model - running champion only")
    
    print("\nTo create a challenger model, run:")
    print("  python ml/auto_retrain.py --force")
    
    print("\n" + "=" * 60)

