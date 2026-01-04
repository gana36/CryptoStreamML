# CryptoStreamML

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?logo=apachekafka)
![InfluxDB](https://img.shields.io/badge/InfluxDB-22ADF6?logo=influxdb&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

Real-time crypto prediction pipeline. Prices stream in every 30s, features calculate on-the-fly, ML predicts direction, results visualize live.
<img width="1280" height="428" alt="image" src="https://github.com/user-attachments/assets/6f5d699c-97ec-4f3c-9874-f651b038837b" />
<img width="1419" height="833" alt="image" src="https://github.com/user-attachments/assets/017b8c42-1695-4e5b-b923-b01702f0571e" />


## Architecture

```
┌─────────────┐     ┌─────────┐     ┌──────────────────┐     ┌──────────┐
│  CoinGecko  │────▶│  Kafka  │────▶│ Stream Processor │────▶│ InfluxDB │
│    API      │     │         │     │                  │     │          │
└─────────────┘     └─────────┘     └────────┬─────────┘     └────┬─────┘
                                             │                    │
                                             ▼                    ▼
┌─────────────┐     ┌─────────┐     ┌──────────────────┐     ┌──────────┐
│ CryptoCompare│────▶│ Training│────▶│   ML Predictor   │     │ Grafana  │
│    API      │     │  Data   │     │  (A/B Testing)   │     │          │
└─────────────┘     └─────────┘     └────────┬─────────┘     └──────────┘
                                             │
                                             ▼
                    ┌─────────┐     ┌──────────────────┐     ┌──────────┐
                    │ MLflow  │◀────│   Auto-Retrain   │◀────│ Evidently│
                    │         │     │                  │     │  (Drift) │
                    └─────────┘     └──────────────────┘     └──────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Ingestion rate | 10 messages/min (5 cryptos × 30s) |
| Processing latency | ~150ms |
| Model accuracy | 59% on 10K samples |
| Buffer for predictions | 25 data points |

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- ~4GB RAM for all services
- Ports 3000, 5000, 8086, 9090, 9092 free

## Setup

```bash
# Start services
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Fetch training data (90 days from CryptoCompare)
python ml/fetch_training_data.py

# Train model
python ml/train_model.py

# Run producer (pulls prices)
python producer/coingecko_producer.py

# Run processor (predictions) - separate terminal
python spark/stream_processor.py
```

## Services

| Port | What |
|------|------|
| 3000 | Grafana (admin/admin) |
| 5000 | MLflow |
| 8086 | InfluxDB |
| 9090 | Prometheus |
| 9092 | Kafka |

## A/B Testing

Two models run simultaneously: champion (80%) and challenger (20%). When drift is detected, auto-retrain kicks in.

```bash
# Check A/B status
python ml/ab_testing.py

# Force retrain challenger
python ml/auto_retrain.py --force

# Drift detection with auto-retrain
python monitoring/drift_detector.py --auto-retrain
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AB_CHAMPION_RATIO` | 0.8 | Traffic to champion model |
| `DRIFT_THRESHOLD` | 0.3 | Drift % to trigger retrain |
| `POLL_INTERVAL` | 30 | Seconds between price fetches |

## Troubleshooting

**"No data" on Grafana**
- Check stream processor is running
- Wait for buffer to reach 25+

**Port already in use**
```bash
netstat -ano | findstr :8000
```

**CoinGecko 429 errors**
- Rate limited. Keep POLL_INTERVAL at 30s

**Prometheus targets DOWN**
- Restart stream processor after code changes

## CI/CD

GitHub Actions on `prod` branch: lint → test → build → deploy

## Structure

```
├── producer/          # CoinGecko → Kafka
├── spark/             # Kafka → ML → InfluxDB
├── ml/                # Training, A/B, auto-retrain
├── monitoring/        # Drift detection
├── models/            # Saved models
├── grafana/           # Dashboard configs
└── prometheus/        # Metrics config
```
.
