# CryptoStreamML

Real-time crypto price streaming pipeline with ML prediction and MLOps.

## 🏗️ Architecture

```
CoinGecko API → Kafka → Stream Processor → InfluxDB → Grafana
                              ↓
                         ML Model
                              ↓
Binance API → Training → MLflow → Model Registry
                              ↓
                    EvidentlyAI (Drift) → Prometheus
```

## 🚀 Quick Start

### 1. Start Infrastructure
```bash
docker-compose up -d
```

Services started:
| Service | Port | Description |
|---------|------|-------------|
| Kafka | 9092 | Message broker |
| InfluxDB | 8086 | Time-series DB |
| Grafana | 3000 | Visualization |
| MLflow | 5000 | Experiment tracking |
| Prometheus | 9090 | Metrics |

### 2. Fetch Training Data (Binance - 90 days hourly)
```bash
pip install -r ml/requirements.txt
python ml/fetch_binance_data.py
```

### 3. Train Model with MLflow
```bash
python ml/train_model.py
```
View experiments: http://localhost:5000

### 4. Start Producer
```bash
pip install -r producer/requirements.txt
python producer/coingecko_producer.py
```

### 5. Start Stream Processor
```bash
pip install -r spark/requirements.txt
python spark/stream_processor.py
```

### 6. View Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090

## 📊 Features

### Streaming Pipeline
- Real-time price ingestion from CoinGecko
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Live ML predictions (UP/DOWN/NEUTRAL)

### MLOps
- **MLflow**: Experiment tracking, model registry
- **Prometheus**: Pipeline metrics (latency, throughput)
- **EvidentlyAI**: Data drift detection

### Technical Indicators
- SMA (5, 15, 24, 50 periods)
- RSI (14 periods)
- MACD + Signal + Histogram
- Bollinger Bands
- ATR, Momentum, Volatility

## 📁 Project Structure

```
CryptoStreamML/
├── docker-compose.yml      # All services
├── producer/               # CoinGecko → Kafka
├── spark/                  # Kafka → InfluxDB
├── ml/
│   ├── fetch_binance_data.py  # Training data
│   └── train_model.py         # MLflow training
├── monitoring/
│   └── drift_detector.py      # EvidentlyAI
├── models/                 # Trained models
├── grafana/               # Dashboards
└── prometheus/            # Metrics config
```

## 🔧 Cryptos Tracked
Bitcoin, Ethereum, Solana, Cardano, Ripple

## 🛑 Shutdown
```bash
docker-compose down
```
