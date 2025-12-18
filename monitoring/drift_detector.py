"""
Real-Time Drift Detection with InfluxDB Production Data
Compares training data (reference) vs real streaming data from InfluxDB.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

import pandas as pd
import numpy as np

try:
    from influxdb_client import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("InfluxDB client not installed. Run: pip install influxdb-client")

try:
    # Evidently v0.7.x - use legacy module for classic API
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    EVIDENTLY_AVAILABLE = False
    print(f"EvidentlyAI import error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# InfluxDB Configuration
INFLUXDB_URL = 'http://localhost:8086'
INFLUXDB_TOKEN = 'my-super-secret-token'
INFLUXDB_ORG = 'crypto-org'
INFLUXDB_BUCKET = 'crypto-prices'

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'data')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
REFERENCE_DATA_PATH = os.path.join(DATA_DIR, 'combined_binance.csv')

# Features to monitor (common between training and production)
MONITORED_FEATURES = [
    'price',
    'rsi',
    'sma_5',
    'volatility_5',
    'momentum_5'
]


def load_reference_data() -> Optional[pd.DataFrame]:
    """Load training data as reference."""
    if not os.path.exists(REFERENCE_DATA_PATH):
        logger.error(f"Reference data not found at {REFERENCE_DATA_PATH}")
        return None
    
    df = pd.read_csv(REFERENCE_DATA_PATH)
    
    # Filter to monitored features that exist
    available = [f for f in MONITORED_FEATURES if f in df.columns]
    df_filtered = df[available].dropna()
    
    logger.info(f"Loaded reference data: {len(df_filtered)} samples, {len(available)} features")
    return df_filtered


def fetch_production_data(minutes: int = 30) -> Optional[pd.DataFrame]:
    """Fetch recent data from InfluxDB."""
    if not INFLUXDB_AVAILABLE:
        logger.error("InfluxDB client not available")
        return None
    
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        query_api = client.query_api()
        
        # Flux query to get recent data
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "crypto_price" or r["_measurement"] == "crypto_indicators")
          |> pivot(rowKey:["_time", "crypto_id"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        
        tables = query_api.query(query)
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                row = {
                    'timestamp': record.get_time(),
                    'crypto_id': record.values.get('crypto_id', ''),
                    'price': record.values.get('price'),
                    'rsi': record.values.get('rsi'),
                    'sma_5': record.values.get('sma_5'),
                    'volatility': record.values.get('volatility'),
                    'momentum': record.values.get('momentum')
                }
                records.append(row)
        
        client.close()
        
        if not records:
            logger.warning("No production data found in InfluxDB")
            return None
        
        df = pd.DataFrame(records)
        
        # Rename to match reference features
        df = df.rename(columns={
            'volatility': 'volatility_5',
            'momentum': 'momentum_5'
        })
        
        # Filter to monitored features
        available = [f for f in MONITORED_FEATURES if f in df.columns]
        df_filtered = df[available].dropna()
        
        logger.info(f"Fetched production data: {len(df_filtered)} samples from last {minutes} minutes")
        return df_filtered
        
    except Exception as e:
        logger.error(f"Failed to fetch from InfluxDB: {e}")
        return None


def detect_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict:
    """
    Run drift detection between reference and production data.
    """
    if not EVIDENTLY_AVAILABLE:
        return {'error': 'EvidentlyAI not installed'}
    
    # Find common columns
    common_cols = list(set(reference_df.columns) & set(production_df.columns))
    
    if len(common_cols) == 0:
        return {'error': 'No common columns between datasets'}
    
    logger.info(f"Comparing {len(common_cols)} features: {common_cols}")
    
    ref_df = reference_df[common_cols].copy()
    prod_df = production_df[common_cols].copy()
    
    # Create drift report using preset
    report = Report(metrics=[DataDriftPreset()])
    
    try:
        report.run(
            reference_data=ref_df,
            current_data=prod_df
        )
        
        # Save HTML report first
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_path = os.path.join(
            REPORTS_DIR,
            f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        report.save_html(report_path)
        
        # Try to get JSON results (API varies by version)
        drift_summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': False,
            'drift_share': 0.0,
            'n_drifted_columns': 0,
            'n_columns': len(common_cols),
            'reference_samples': len(ref_df),
            'production_samples': len(prod_df),
            'drifted_columns': [],
            'report_path': report_path
        }
        
        # Try different API methods to get results
        try:
            # v0.7.x uses json()
            result = report.json()
            import json
            data = json.loads(result)
            for metric in data.get('metrics', []):
                metric_result = metric.get('result', {})
                if 'drift_share' in metric_result:
                    drift_summary['drift_share'] = metric_result.get('drift_share', 0)
                    drift_summary['dataset_drift_detected'] = metric_result.get('dataset_drift', False)
                    drift_summary['n_drifted_columns'] = metric_result.get('number_of_drifted_columns', 0)
                if 'drift_by_columns' in metric_result:
                    for col, col_data in metric_result['drift_by_columns'].items():
                        if col_data.get('drift_detected', False):
                            drift_summary['drifted_columns'].append({
                                'column': col,
                                'drift_score': col_data.get('drift_score', 0),
                                'stattest': col_data.get('stattest_name', 'ks')
                            })
        except Exception:
            # If JSON parsing fails, just report that HTML was generated
            logger.info("Could not parse JSON, HTML report was generated")
        
        return drift_summary
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Run drift detection with real InfluxDB data."""
    print("\n" + "="*60)
    print("REAL-TIME DRIFT DETECTION")
    print("Reference: Training Data | Current: InfluxDB Production Data")
    print("="*60 + "\n")
    
    if not EVIDENTLY_AVAILABLE:
        print("ERROR: EvidentlyAI not installed. Run: pip install evidently")
        return
    
    # Load reference data
    print("Loading reference (training) data...")
    reference_df = load_reference_data()
    
    if reference_df is None:
        print("ERROR: Could not load reference data. Run fetch_binance_data.py first.")
        return
    
    # Fetch production data from InfluxDB
    print("\nFetching production data from InfluxDB...")
    production_df = fetch_production_data(minutes=60)
    
    if production_df is None or len(production_df) < 10:
        print("WARNING: Not enough production data in InfluxDB.")
        print("Let the stream processor run for a few more minutes, then try again.")
        print("\nUsing simulated production data for demo...")
        
        # Create simulated data with slight drift
        np.random.seed(42)
        n = 100
        production_df = pd.DataFrame({
            'price': np.random.normal(90000, 5000, n),  # Higher price (drift)
            'rsi': np.random.uniform(20, 80, n),
            'sma_5': np.random.normal(88000, 4000, n),
            'volatility_5': np.random.exponential(0.025, n),  # Higher volatility
            'momentum_5': np.random.normal(600, 2000, n)
        })
        print(f"Generated {len(production_df)} simulated samples with drift")
    
    # Run drift detection
    print("\nRunning drift detection...")
    result = detect_drift(reference_df, production_df)
    
    # Print results
    print("\n" + "="*60)
    print("DRIFT DETECTION RESULTS")
    print("="*60)
    
    if 'error' in result:
        print(f"ERROR: {result['error']}")
        return
    
    print(f"\n📊 Dataset Drift Detected: {'🔴 YES' if result['dataset_drift_detected'] else '🟢 NO'}")
    print(f"📈 Drift Share: {result['drift_share']:.1%}")
    print(f"📋 Drifted Features: {result['n_drifted_columns']} / {result['n_columns']}")
    print(f"\n📁 Reference samples: {result['reference_samples']:,}")
    print(f"📁 Production samples: {result['production_samples']:,}")
    
    if result['drifted_columns']:
        print("\n⚠️  Features with Drift:")
        for col in result['drifted_columns']:
            print(f"   - {col['column']}: score={col['drift_score']:.4f} (test: {col['stattest']})")
    else:
        print("\n✅ No significant feature drift detected!")
    
    if 'report_path' in result:
        print(f"\n📄 Full HTML Report: {result['report_path']}")
        print("   Open in browser to see detailed visualizations")
    
    # Check for auto-retrain
    drift_share = result.get('drift_share', 0)
    auto_retrain = '--auto-retrain' in sys.argv or '--retrain' in sys.argv
    
    if auto_retrain and drift_share > 0:
        print("\n" + "="*60)
        print("AUTO-RETRAIN CHECK")
        print("="*60)
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from ml.auto_retrain import check_and_retrain
            
            retrain_result = check_and_retrain(drift_share)
            
            if retrain_result.get('retrain_triggered'):
                if retrain_result.get('retrain_result', {}).get('success'):
                    print("\n🔄 Auto-retrain triggered and successful!")
                    print("   New challenger model ready for A/B testing.")
                else:
                    print("\n❌ Auto-retrain triggered but failed.")
        except Exception as e:
            print(f"\n⚠️ Auto-retrain check failed: {e}")
    
    print("\n" + "="*60)
    print("\nTip: Run with --auto-retrain to enable automatic retraining on drift")


if __name__ == '__main__':
    import sys
    main()

