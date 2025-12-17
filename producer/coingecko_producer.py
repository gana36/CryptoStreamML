"""
CoinGecko API Producer for Kafka
Fetches real-time cryptocurrency prices and publishes to Kafka topic.
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'crypto-prices'
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'

# Cryptocurrencies to track
CRYPTO_IDS = ['bitcoin', 'ethereum', 'solana', 'cardano', 'ripple']
VS_CURRENCY = 'usd'

# Polling interval (seconds) - respects CoinGecko rate limits
POLL_INTERVAL = 30


def fetch_prices() -> Optional[List[Dict]]:
    """
    Fetch current prices from CoinGecko API.
    Returns list of price data dictionaries.
    """
    try:
        # Get simple price with additional market data
        params = {
            'ids': ','.join(CRYPTO_IDS),
            'vs_currencies': VS_CURRENCY,
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        response = requests.get(
            f'{COINGECKO_API_URL}/simple/price',
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Transform to list of records
        records = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        for crypto_id, values in data.items():
            record = {
                'crypto_id': crypto_id,
                'symbol': crypto_id[:3].upper(),  # Approximate symbol
                'price_usd': values.get(VS_CURRENCY, 0),
                'market_cap': values.get(f'{VS_CURRENCY}_market_cap', 0),
                'volume_24h': values.get(f'{VS_CURRENCY}_24h_vol', 0),
                'change_24h_percent': values.get(f'{VS_CURRENCY}_24h_change', 0),
                'last_updated': values.get('last_updated_at', 0),
                'fetch_timestamp': timestamp
            }
            records.append(record)
        
        logger.info(f"Fetched prices for {len(records)} cryptocurrencies")
        return records
        
    except requests.RequestException as e:
        logger.error(f"Error fetching prices: {e}")
        return None


def create_kafka_producer() -> KafkaProducer:
    """
    Create and configure Kafka producer.
    """
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None,
        acks='all',
        retries=3,
        retry_backoff_ms=1000
    )


def on_send_success(record_metadata):
    """Callback for successful message delivery."""
    logger.debug(
        f"Message delivered to {record_metadata.topic} "
        f"[partition {record_metadata.partition}] "
        f"@ offset {record_metadata.offset}"
    )


def on_send_error(excp):
    """Callback for failed message delivery."""
    logger.error(f"Message delivery failed: {excp}")


def publish_to_kafka(producer: KafkaProducer, records: List[Dict]) -> int:
    """
    Publish price records to Kafka topic.
    Returns number of messages sent.
    """
    sent_count = 0
    
    for record in records:
        try:
            # Use crypto_id as key for partitioning
            future = producer.send(
                KAFKA_TOPIC,
                key=record['crypto_id'],
                value=record
            )
            future.add_callback(on_send_success)
            future.add_errback(on_send_error)
            sent_count += 1
            
        except KafkaError as e:
            logger.error(f"Failed to send message for {record['crypto_id']}: {e}")
    
    # Ensure all messages are sent
    producer.flush()
    return sent_count


def run_producer():
    """
    Main producer loop - fetches prices and publishes to Kafka continuously.
    """
    logger.info("Starting CoinGecko to Kafka producer...")
    logger.info(f"Tracking: {', '.join(CRYPTO_IDS)}")
    logger.info(f"Kafka topic: {KAFKA_TOPIC}")
    logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
    
    producer = None
    
    try:
        producer = create_kafka_producer()
        logger.info("Connected to Kafka successfully")
        
        while True:
            # Fetch prices from CoinGecko
            records = fetch_prices()
            
            if records:
                # Publish to Kafka
                sent = publish_to_kafka(producer, records)
                logger.info(f"Published {sent} messages to Kafka")
                
                # Log sample data
                for record in records[:2]:
                    logger.info(
                        f"  {record['crypto_id']}: ${record['price_usd']:,.2f} "
                        f"({record['change_24h_percent']:+.2f}%)"
                    )
            
            # Wait before next poll
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Shutting down producer...")
    except Exception as e:
        logger.error(f"Producer error: {e}")
        raise
    finally:
        if producer:
            producer.close()
            logger.info("Kafka producer closed")


if __name__ == '__main__':
    run_producer()
