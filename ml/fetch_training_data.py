"""
Fetch Historical Crypto Data from CryptoCompare API
Works globally (no regional restrictions like Binance).
FREE API - no authentication required.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CRYPTOCOMPARE_API_URL = 'https://min-api.cryptocompare.com/data/v2'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Cryptocurrencies to fetch
CRYPTO_SYMBOLS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'solana': 'SOL',
    'cardano': 'ADA',
    'ripple': 'XRP'
}

# Fetch 90 days of hourly data = ~2160 candles per symbol
DAYS = 90


def fetch_hourly_ohlcv(symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch historical hourly OHLCV data from CryptoCompare.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC')
        days: Number of days to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # CryptoCompare limit is 2000 per request, so we can get all at once
        limit = min(days * 24, 2000)
        
        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': limit
        }
        
        response = requests.get(
            f'{CRYPTOCOMPARE_API_URL}/histohour',
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('Response') == 'Error':
            logger.error(f"API Error for {symbol}: {data.get('Message')}")
            return None
        
        # Parse data
        ohlcv_data = data.get('Data', {}).get('Data', [])
        
        if not ohlcv_data:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        df = pd.DataFrame(ohlcv_data)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to standard format
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumefrom': 'volume',
            'volumeto': 'quote_volume'
        })
        
        # Use close as primary price
        df['price'] = df['close']
        
        # Keep relevant columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'price', 'volume', 'quote_volume']]
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
        
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for model training.
    """
    df = df.copy()
    
    # Price changes
    df['price_change'] = df['price'].pct_change()
    df['price_change_1h'] = df['price'].pct_change(periods=1)
    df['price_change_4h'] = df['price'].pct_change(periods=4)
    df['price_change_24h'] = df['price'].pct_change(periods=24)
    
    # Simple Moving Averages
    df['sma_5'] = df['price'].rolling(window=5).mean()
    df['sma_15'] = df['price'].rolling(window=15).mean()
    df['sma_24'] = df['price'].rolling(window=24).mean()
    df['sma_50'] = df['price'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['price'].ewm(span=12).mean()
    df['ema_26'] = df['price'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # SMA ratios (normalized features)
    df['sma_ratio_5_15'] = df['sma_5'] / df['sma_15']
    df['sma_ratio_5_24'] = df['sma_5'] / df['sma_24']
    df['price_to_sma_5'] = df['price'] / df['sma_5']
    df['price_to_sma_50'] = df['price'] / df['sma_50']
    
    # RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    df['bb_std'] = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volatility
    df['volatility_5'] = df['price_change'].rolling(window=5).std()
    df['volatility_24'] = df['price_change'].rolling(window=24).std()
    
    # Volume features
    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_5']
    df['volume_change'] = df['volume'].pct_change()
    
    # Momentum
    df['momentum_5'] = df['price'] - df['price'].shift(5)
    df['momentum_12'] = df['price'] - df['price'].shift(12)
    df['momentum_24'] = df['price'] - df['price'].shift(24)
    
    # Rate of Change (ROC)
    df['roc_5'] = (df['price'] / df['price'].shift(5) - 1) * 100
    df['roc_12'] = (df['price'] / df['price'].shift(12) - 1) * 100
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Target variable: price direction in next period
    # 1 = UP (>0.5%), 0 = NEUTRAL, -1 = DOWN (<-0.5%)
    future_change = df['price'].shift(-1) / df['price'] - 1
    df['target'] = pd.cut(
        future_change,
        bins=[-float('inf'), -0.005, 0.005, float('inf')],
        labels=[-1, 0, 1]
    ).astype(float)
    
    return df


def main():
    """
    Main function to fetch and save historical data from CryptoCompare.
    """
    print("\n" + "="*60)
    print("CRYPTOCOMPARE DATA FETCHER")
    print("Free API - Works globally (no regional restrictions)")
    print("="*60 + "\n")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_data = []
    
    for crypto_name, symbol in CRYPTO_SYMBOLS.items():
        logger.info(f"Fetching {crypto_name} ({symbol})...")
        
        df = fetch_hourly_ohlcv(symbol, DAYS)
        
        if df is not None and len(df) > 0:
            # Add crypto identifier
            df['crypto_id'] = crypto_name
            df['symbol'] = symbol
            
            # Calculate features
            df = calculate_features(df)
            all_data.append(df)
            
            # Save individual file
            filepath = os.path.join(DATA_DIR, f'{crypto_name}_cryptocompare.csv')
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {filepath} ({len(df)} records)")
        
        # Rate limiting
        time.sleep(1)
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save as the standard training file name
        combined_path = os.path.join(DATA_DIR, 'combined_binance.csv')
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined data to {combined_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Source: CryptoCompare API")
        print(f"Cryptocurrencies: {', '.join(CRYPTO_SYMBOLS.keys())}")
        print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"Total records: {len(combined_df)}")
        print(f"Candle interval: 1 hour")
        
        print(f"\nFeatures calculated:")
        print("  - SMA (5, 15, 24, 50 periods)")
        print("  - EMA (12, 26 periods)")
        print("  - MACD + Signal + Histogram")
        print("  - RSI (14 periods)")
        print("  - Bollinger Bands (20 periods)")
        print("  - ATR (14 periods)")
        print("  - Volatility (5, 24 periods)")
        print("  - Momentum (5, 12, 24 periods)")
        print("  - Volume ratios")
        
        # Target distribution
        valid_targets = combined_df.dropna(subset=['target'])
        print(f"\nTarget distribution:")
        for target_val in sorted(valid_targets['target'].unique()):
            count = len(valid_targets[valid_targets['target'] == target_val])
            label = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}.get(int(target_val), 'UNKNOWN')
            print(f"  {label}: {count} ({count/len(valid_targets)*100:.1f}%)")
        
        # Data per crypto
        print(f"\nRecords per cryptocurrency:")
        for crypto in CRYPTO_SYMBOLS.keys():
            count = len(combined_df[combined_df['crypto_id'] == crypto])
            print(f"  {crypto}: {count}")
    else:
        print("\nNo data fetched. Check your internet connection.")


if __name__ == '__main__':
    main()
