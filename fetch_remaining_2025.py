import pandas as pd
import requests
import io
import argparse
import sys
import db_utils

# AlphaVantage Configuration
SYMBOL_FROM = "USD"
SYMBOL_TO = "JPY"
FUNCTION = "FX_INTRADAY"
INTERVAL = "1min"
OUTPUTSIZE = "full" 

def fetch_and_update(api_key):
    conn = db_utils.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM market_data")
    last_ts_str = cursor.fetchone()[0]
    conn.close()
    
    if last_ts_str:
        last_ts = pd.to_datetime(last_ts_str).tz_localize(None)
        print(f"Database last timestamp: {last_ts}")
    else:
        print("Database is empty. Fetching all available.")
        last_ts = pd.Timestamp.min

    url = f"https://www.alphavantage.co/query?function={FUNCTION}&from_symbol={SYMBOL_FROM}&to_symbol={SYMBOL_TO}&interval={INTERVAL}&outputsize={OUTPUTSIZE}&apikey={api_key}&datatype=csv"
    print(f"Requesting {url.replace(api_key, 'HIDDEN')}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Check raw text first
        if "{" in response.text and "}" in response.text and "Information" in response.text:
             print("API returned JSON Message:")
             print(response.text)
             return

        df = pd.read_csv(io.BytesIO(response.content))
        print(f"Fetched {len(df)} rows.")
        
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        
        if 'timestamp' not in df.columns:
            print("Error: 'timestamp' column missing.")
            print("Columns found:", df.columns.tolist())
            return
            
        df.rename(columns={'timestamp': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['ts_naive'] = df['timestamp'].dt.tz_localize(None)
        
        new_data = df[df['ts_naive'] > last_ts].copy()
        
        if len(new_data) == 0:
            print("No new data found.")
            return
            
        print(f"Found {len(new_data)} new rows to insert.")
        
        conn = db_utils.get_db_connection()
        if 'volume' not in new_data.columns:
            new_data['volume'] = 0
            
        final_df = new_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        final_df.to_sql('market_data', conn, if_exists='append', index=False)
        print("Successfully inserted new data.")
        conn.close()
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="AlphaVantage API Key")
    args = parser.parse_args()
    
    fetch_and_update(args.api_key)
