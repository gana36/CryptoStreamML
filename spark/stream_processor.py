"""
Spark Structured Streaming Processor
Consumes crypto prices from Kafka, calculates indicators, runs ML predictions, writes to InfluxDB.
Includes Prometheus metrics for observability.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Optional
import threading

import joblib
import numpy as np
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'crypto-prices'
KAFKA_GROUP_ID = 'crypto-processor'

# InfluxDB Configuration
INFLUXDB_URL = 'http://localhost:8086'
INFLUXDB_TOKEN = 'my-super-secret-token'
INFLUXDB_ORG = 'crypto-org'
INFLUXDB_BUCKET = 'crypto-prices'

# Prometheus Configuration
PROMETHEUS_PORT = 8000

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    MESSAGES_PROCESSED = Counter(
        'crypto_messages_processed_total',
        'Total number of messages processed',
        ['crypto_id']
    )
    PROCESSING_LATENCY = Histogram(
        'crypto_processing_latency_seconds',
        'Message processing latency',
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    PREDICTION_COUNTER = Counter(
        'crypto_predictions_total',
        'Predictions made by direction',
        ['crypto_id', 'prediction']
    )
    CURRENT_PRICE = Gauge(
        'crypto_current_price_usd',
        'Current price in USD',
        ['crypto_id']
    )
    RSI_VALUE = Gauge(
        'crypto_rsi_value',
        'Current RSI value',
        ['crypto_id']
    )
    BUFFER_SIZE = Gauge(
        'crypto_buffer_size',
        'Price buffer size per crypto',
        ['crypto_id']
    )


class PriceBuffer:
    """
    Rolling buffer to store price history for indicator calculations.
    Maintains separate buffers per cryptocurrency.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffers: Dict[str, deque] = {}
    
    def add(self, crypto_id: str, price: float, timestamp: float):
        if crypto_id not in self.buffers:
            self.buffers[crypto_id] = deque(maxlen=self.max_size)
        
        self.buffers[crypto_id].append({
            'price': price,
            'timestamp': timestamp
        })
    
    def get_prices(self, crypto_id: str, n: int = None) -> list:
        if crypto_id not in self.buffers:
            return []
        
        buffer = list(self.buffers[crypto_id])
        if n:
            return [p['price'] for p in buffer[-n:]]
        return [p['price'] for p in buffer]
    
    def get_count(self, crypto_id: str) -> int:
        return len(self.buffers.get(crypto_id, []))


class TechnicalIndicators:
    """
    Calculate technical indicators from price data.
    """
    
    @staticmethod
    def sma(prices: list, period: int) -> Optional[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])
    
    @staticmethod
    def rsi(prices: list, period: int = 14) -> Optional[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def momentum(prices: list, period: int) -> Optional[float]:
        """Price momentum"""
        if len(prices) < period + 1:
            return None
        return prices[-1] - prices[-(period+1)]
    
    @staticmethod
    def volatility(prices: list, period: int) -> Optional[float]:
        """Rolling standard deviation of returns"""
        if len(prices) < period + 1:
            return None
        
        returns = np.diff(prices[-(period+1):]) / prices[-(period+1):-1]
        return np.std(returns)


class MLPredictor:
    """
    Load and run ML model for price direction prediction.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and scaler"""
        model_path = os.path.join(MODELS_DIR, 'price_predictor.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
        config_path = os.path.join(MODELS_DIR, 'model_config.pkl')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Predictions disabled.")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        if os.path.exists(config_path):
            self.config = joblib.load(config_path)
            logger.info(f"Loaded config from {config_path}")
    
    def predict(self, features: Dict) -> Optional[Dict]:
        """
        Make prediction from calculated features.
        Returns prediction label and probability.
        """
        if self.model is None or self.scaler is None:
            return None
        
        try:
            # Extract feature values in correct order
            feature_cols = self.config['feature_columns']
            feature_values = []
            
            for col in feature_cols:
                val = features.get(col)
                if val is None:
                    return None  # Missing feature
                feature_values.append(val)
            
            # Scale and predict
            X = np.array([feature_values])
            X_scaled = self.scaler.transform(X)
            
            prediction = int(self.model.predict(X_scaled)[0])
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Map prediction to label
            label = self.config['target_mapping'].get(prediction, 'UNKNOWN')
            
            return {
                'prediction': prediction,
                'label': label,
                'probability': float(max(probabilities)),
                'probabilities': {
                    'DOWN': float(probabilities[0]) if len(probabilities) > 0 else 0,
                    'NEUTRAL': float(probabilities[1]) if len(probabilities) > 1 else 0,
                    'UP': float(probabilities[2]) if len(probabilities) > 2 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None


class StreamProcessor:
    """
    Main stream processing class.
    """
    
    def __init__(self):
        self.price_buffer = PriceBuffer(max_size=100)
        self.indicators = TechnicalIndicators()
        self.predictor = MLPredictor()
        self.influx_client = None
        self.write_api = None
        self._connect_influxdb()
    
    def _connect_influxdb(self):
        """Connect to InfluxDB"""
        try:
            self.influx_client = InfluxDBClient(
                url=INFLUXDB_URL,
                token=INFLUXDB_TOKEN,
                org=INFLUXDB_ORG
            )
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            logger.info("Connected to InfluxDB")
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
    
    def calculate_features(self, crypto_id: str, current_price: float) -> Dict:
        """
        Calculate all technical indicators and features.
        """
        prices = self.price_buffer.get_prices(crypto_id)
        
        if not prices:
            return {}
        
        features = {
            'current_price': current_price,
            'data_points': len(prices)
        }
        
        # SMAs
        sma_5 = self.indicators.sma(prices, 5)
        sma_15 = self.indicators.sma(prices, 15)
        sma_24 = self.indicators.sma(prices, 24)
        sma_50 = self.indicators.sma(prices, 50) if len(prices) >= 50 else sma_24
        
        features['sma_5'] = sma_5
        features['sma_15'] = sma_15
        features['sma_24'] = sma_24
        features['sma_50'] = sma_50
        
        # SMA ratios for ML
        if sma_5 and sma_15:
            features['sma_ratio_5_15'] = sma_5 / sma_15
        if sma_5 and sma_24:
            features['sma_ratio_5_24'] = sma_5 / sma_24
        if sma_5:
            features['price_to_sma_5'] = current_price / sma_5
        if sma_50:
            features['price_to_sma_50'] = current_price / sma_50
        
        # RSI
        features['rsi'] = self.indicators.rsi(prices, 14)
        
        # Momentum
        features['momentum_5'] = self.indicators.momentum(prices, 5)
        features['momentum_12'] = self.indicators.momentum(prices, 12)
        features['momentum_24'] = self.indicators.momentum(prices, 24) if len(prices) >= 24 else features.get('momentum_12', 0)
        
        # Volatility
        features['volatility_5'] = self.indicators.volatility(prices, 5)
        features['volatility_24'] = self.indicators.volatility(prices, 24)
        
        # Price changes (approximated from buffer)
        if len(prices) >= 2:
            features['price_change_1h'] = (current_price - prices[-2]) / prices[-2]
        if len(prices) >= 5:
            features['price_change_4h'] = (current_price - prices[-5]) / prices[-5]
        if len(prices) >= 25:
            features['price_change_24h'] = (current_price - prices[-25]) / prices[-25]
        
        # Rate of Change (ROC)
        if len(prices) >= 5:
            features['roc_5'] = ((current_price / prices[-5]) - 1) * 100
        if len(prices) >= 12:
            features['roc_12'] = ((current_price / prices[-12]) - 1) * 100
        
        # MACD - simplified calculation
        if len(prices) >= 26:
            ema_12 = self._ema(prices, 12)
            ema_26 = self._ema(prices, 26)
            macd = ema_12 - ema_26
            features['macd'] = macd
            features['macd_signal'] = macd * 0.8  # Simplified
            features['macd_histogram'] = macd - features['macd_signal']
        else:
            features['macd'] = 0
            features['macd_signal'] = 0
            features['macd_histogram'] = 0
        
        # Bollinger Bands position
        if len(prices) >= 20:
            bb_middle = self.indicators.sma(prices, 20)
            bb_std = np.std(prices[-20:])
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            if bb_upper != bb_lower:
                features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                features['bb_position'] = 0.5
        else:
            features['bb_position'] = 0.5
        
        # ATR - simplified (using price range)
        if len(prices) >= 14:
            true_ranges = []
            for i in range(-14, -1):
                high_low = max(prices[i], prices[i+1]) - min(prices[i], prices[i+1])
                true_ranges.append(high_low)
            features['atr'] = np.mean(true_ranges) if true_ranges else 0
        else:
            features['atr'] = 0
        
        # Volume ratio (placeholder - would need volume buffer)
        features['volume_ratio'] = 1.0
        
        return features
    
    def _ema(self, prices: list, window: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < window:
            return prices[-1] if prices else 0
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()
        return np.dot(prices[-window:], weights)
    
    def write_to_influxdb(self, crypto_id: str, data: Dict, prediction: Optional[Dict]):
        """
        Write processed data to InfluxDB.
        """
        if not self.write_api:
            return
        
        try:
            # Main price point
            point = Point("crypto_price") \
                .tag("crypto_id", crypto_id) \
                .field("price", float(data.get('current_price', 0))) \
                .field("market_cap", float(data.get('market_cap', 0))) \
                .field("volume_24h", float(data.get('volume_24h', 0))) \
                .field("change_24h", float(data.get('change_24h_percent', 0)))
            
            # Add indicators if available
            if data.get('sma_5'):
                point = point.field("sma_5", float(data['sma_5']))
            if data.get('sma_15'):
                point = point.field("sma_15", float(data['sma_15']))
            if data.get('rsi'):
                point = point.field("rsi", float(data['rsi']))
            
            # Add prediction if available
            if prediction:
                point = point.field("prediction", prediction['prediction'])
                point = point.field("prediction_label", prediction['label'])
                point = point.field("prediction_confidence", prediction['probability'])
            
            point = point.time(datetime.now(timezone.utc), WritePrecision.MS)
            
            self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
            logger.debug(f"Written to InfluxDB: {crypto_id}")
            
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")
    
    def process_message(self, message: Dict):
        """
        Process a single message from Kafka.
        """
        crypto_id = message.get('crypto_id')
        price = message.get('price_usd')
        timestamp = message.get('last_updated', 0)
        
        if not crypto_id or not price:
            logger.warning(f"Invalid message: {message}")
            return
        
        # Add to buffer
        self.price_buffer.add(crypto_id, price, timestamp)
        
        # Calculate features
        features = self.calculate_features(crypto_id, price)
        
        # Merge original message data
        features.update({
            'market_cap': message.get('market_cap', 0),
            'volume_24h': message.get('volume_24h', 0),
            'change_24h_percent': message.get('change_24h_percent', 0)
        })
        
        # Run ML prediction
        prediction = self.predictor.predict(features)
        
        # Write to InfluxDB
        self.write_to_influxdb(crypto_id, features, prediction)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            MESSAGES_PROCESSED.labels(crypto_id=crypto_id).inc()
            CURRENT_PRICE.labels(crypto_id=crypto_id).set(price)
            
            if features.get('rsi') is not None:
                RSI_VALUE.labels(crypto_id=crypto_id).set(features['rsi'])
            
            BUFFER_SIZE.labels(crypto_id=crypto_id).set(
                self.price_buffer.get_count(crypto_id)
            )
            
            if prediction:
                PREDICTION_COUNTER.labels(
                    crypto_id=crypto_id, 
                    prediction=prediction['label']
                ).inc()
        
        # Log summary
        buffer_size = self.price_buffer.get_count(crypto_id)
        pred_str = f" | Prediction: {prediction['label']} ({prediction['probability']:.1%})" if prediction else ""
        
        # Format indicator values safely
        sma_5_str = f"{features['sma_5']:,.2f}" if features.get('sma_5') is not None else "N/A"
        rsi_str = f"{features['rsi']:.1f}" if features.get('rsi') is not None else "N/A"
        
        logger.info(
            f"{crypto_id}: ${price:,.2f} | "
            f"SMA5: {sma_5_str} | "
            f"RSI: {rsi_str} | "
            f"Buffer: {buffer_size}{pred_str}"
        )
    
    def run(self):
        """
        Main processing loop.
        """
        logger.info("Starting stream processor...")
        logger.info(f"Kafka topic: {KAFKA_TOPIC}")
        logger.info(f"InfluxDB bucket: {INFLUXDB_BUCKET}")
        
        # Start Prometheus metrics server
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(PROMETHEUS_PORT)
                logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
            except Exception as e:
                logger.warning(f"Could not start Prometheus server: {e}")
        
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        logger.info("Connected to Kafka. Waiting for messages...")
        
        try:
            for message in consumer:
                start_time = time.time()
                self.process_message(message.value)
                
                # Record processing latency
                if PROMETHEUS_AVAILABLE:
                    PROCESSING_LATENCY.observe(time.time() - start_time)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            consumer.close()
            if self.influx_client:
                self.influx_client.close()
            logger.info("Stream processor stopped")


def main():
    processor = StreamProcessor()
    processor.run()


if __name__ == '__main__':
    main()
