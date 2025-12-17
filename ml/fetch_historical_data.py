"""
Fetch Historical Crypto Data from CoinGecko
Downloads OHLC data for model training.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Cryptocurrencies to fetch
CRYPTO_IDS = ['bitcoin', 'ethereum', 'solana']
VS_CURRENCY = 'usd'
DAYS = 90  # Fetch 90 days of data


def fetch_market_chart(crypto_id: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLC data from CoinGecko (free endpoint).
    Returns DataFrame with timestamp, open, high, low, close prices.
    
    Note: Free tier supports days: 1, 7, 14, 30, 90, 180, 365
    """
    try:
        # Use OHLC endpoint (free, no auth required)
        # Valid days values: 1, 7, 14, 30, 90, 180, 365
        response = requests.get(
            f'{COINGECKO_API_URL}/coins/{crypto_id}/ohlc',
            params={
                'vs_currency': VS_CURRENCY,
                'days': days
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # OHLC format: [[timestamp, open, high, low, close], ...]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Use close price as the primary price
        df['price'] = df['close']
        
        # Estimate volume and market cap (not available in OHLC, use placeholders)
        df['volume'] = 0  # Placeholder
        df['market_cap'] = 0  # Placeholder
        
        # Add crypto identifier
        df['crypto_id'] = crypto_id
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Fetched {len(df)} data points for {crypto_id}")
        return df
        
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {crypto_id}: {e}")
        return None


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for model training.
    Uses smaller windows suitable for limited data (OHLC ~23 points for 90 days).
    """
    df = df.copy()
    
    # Price changes
    df['price_change'] = df['price'].pct_change()
    df['price_change_1h'] = df['price'].pct_change(periods=1)
    df['price_change_4h'] = df['price'].pct_change(periods=2)  # Approximate with 2 periods
    df['price_change_24h'] = df['price'].pct_change(periods=3)  # Approximate with 3 periods
    
    # Simple Moving Averages (smaller windows for limited data)
    df['sma_5'] = df['price'].rolling(window=3, min_periods=2).mean()
    df['sma_15'] = df['price'].rolling(window=5, min_periods=3).mean()
    df['sma_24'] = df['price'].rolling(window=7, min_periods=4).mean()
    
    # SMA ratios (normalized features)
    df['sma_ratio_5_15'] = df['sma_5'] / df['sma_15']
    df['sma_ratio_5_24'] = df['sma_5'] / df['sma_24']
    df['price_to_sma_5'] = df['price'] / df['sma_5']
    
    # RSI (Relative Strength Index) - use smaller window
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5, min_periods=3).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5, min_periods=3).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Default to neutral RSI
    
    # Volatility (smaller windows)
    df['volatility_5'] = df['price_change'].rolling(window=3, min_periods=2).std()
    df['volatility_24'] = df['price_change'].rolling(window=5, min_periods=3).std()
    
    # Fill volatility NaN with small default
    df['volatility_5'] = df['volatility_5'].fillna(0.01)
    df['volatility_24'] = df['volatility_24'].fillna(0.01)
    
    # Volume ratio (placeholder since OHLC doesn't have volume)
    df['volume_ratio'] = 1.0
    
    # Momentum (smaller windows)
    df['momentum_5'] = df['price'] - df['price'].shift(2)
    df['momentum_12'] = df['price'] - df['price'].shift(4)
    
    # Fill momentum NaN with 0
    df['momentum_5'] = df['momentum_5'].fillna(0)
    df['momentum_12'] = df['momentum_12'].fillna(0)
    
    # Target variable: price direction in next period
    # 1 = UP (>1%), 0 = NEUTRAL, -1 = DOWN (<-1%)
    future_change = df['price'].shift(-1) / df['price'] - 1
    df['target'] = pd.cut(
        future_change,
        bins=[-float('inf'), -0.01, 0.01, float('inf')],
        labels=[-1, 0, 1]
    ).astype(float)
    
    return df


def main():
    """
    Main function to fetch and save historical data.
    """
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_data = []
    
    for crypto_id in CRYPTO_IDS:
        logger.info(f"Fetching data for {crypto_id}...")
        
        df = fetch_market_chart(crypto_id, DAYS)
        
        if df is not None:
            # Calculate features
            df = calculate_features(df)
            all_data.append(df)
            
            # Save individual file
            filepath = os.path.join(DATA_DIR, f'{crypto_id}_historical.csv')
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {filepath}")
        
        # Rate limiting - wait longer between requests to avoid 429 errors
        time.sleep(10)
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(DATA_DIR, 'combined_historical.csv')
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined data to {combined_path}")
        logger.info(f"Total records: {len(combined_df)}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Cryptocurrencies: {', '.join(CRYPTO_IDS)}")
        print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"Total records: {len(combined_df)}")
        print(f"\nFeatures calculated:")
        print("  - SMA (5, 15, 24 periods)")
        print("  - RSI (14 periods)")
        print("  - Volatility (5, 24 periods)")
        print("  - Momentum (5, 12 periods)")
        print("  - Volume ratios")
        print(f"\nTarget distribution:")
        print(combined_df['target'].value_counts().sort_index())


if __name__ == '__main__':
    main()
