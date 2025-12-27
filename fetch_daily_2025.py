import pandas as pd
import requests
import io
import argparse
import sys
import db_utils

# AlphaVantage Configuration
SYMBOL_FROM = "USD"
SYMBOL_TO = "JPY"
FUNCTION = "FX_DAILY"
OUTPUTSIZE = "compact" # Last 100 days

def fetch_and_update(api_key):
    # 1. Get Last Date from DB
    conn = db_utils.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM market_data")
    last_ts_str = cursor.fetchone()[0]
    conn.close()
    
    if last_ts_str:
        last_ts = pd.to_datetime(last_ts_str).tz_localize(None)
        print(f"Database last timestamp: {last_ts}")
    else:
        print("Database is empty.")
        last_ts = pd.Timestamp.min

    print(f"Fetching Daily Data (Last 100 Days) to fill gap...")
    url = f"https://www.alphavantage.co/query?function={FUNCTION}&from_symbol={SYMBOL_FROM}&to_symbol={SYMBOL_TO}&outputsize={OUTPUTSIZE}&apikey={api_key}&datatype=csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        if "Error Message" in response.text or "Information" in response.text:
            print("API Message:", response.text)
            return

        df = pd.read_csv(io.BytesIO(response.content))
        print(f"Fetched {len(df)} rows.")
        
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={'timestamp': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['ts_naive'] = df['timestamp'].dt.tz_localize(None)
        
        # Filter > last_ts
        # Note: last_ts might be intraday (e.g. July 1 18:29).
        # Daily data will be July 2 00:00, July 3 00:00...
        new_data = df[df['ts_naive'] > last_ts].copy()
        
        if len(new_data) == 0:
            print("No new data found.")
            return
            
        print(f"Found {len(new_data)} new daily rows to insert (Oct/Nov/Dec).")
        
        conn = db_utils.get_db_connection()
        if 'volume' not in new_data.columns:
            new_data['volume'] = 0
            
        final_df = new_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        final_df.to_sql('market_data', conn, if_exists='append', index=False)
        print("Successfully inserted new DAILY data.")
        conn.close()
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="AlphaVantage API Key")
    args = parser.parse_args()
    
    fetch_and_update(args.api_key)
