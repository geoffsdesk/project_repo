import pandas as pd
import requests
import io
import os

# Configuration
API_KEY = "YOUR_ALPHAVANTAGE_API_KEY" # TODO: Replace with your actual key or use os.getenv("ALPHA_VANTAGE_KEY")
SYMBOL_FROM = "USD"
SYMBOL_TO = "JPY"
INTERVAL = "DAILY" # Not used for FX_DAILY but keeping variable
OUTPUTSIZE = "full"
EXISTING_DATA_PATH = r"C:\Users\geoff\.gemini\antigravity\scratch\project_repo\USDJPY 1M_CANDLESTICK DATA 2015-2025\USD_JPY_2015_07_2025_BID.csv"
OUTPUT_FILENAME = "USD_JPY_DAILY_AUGMENTED.csv"

def fetch_alphavantage_data():
    print(f"Fetching data from AlphaVantage for {SYMBOL_FROM}/{SYMBOL_TO}...")
    # FX_DAILY does not take 'interval'
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={SYMBOL_FROM}&to_symbol={SYMBOL_TO}&outputsize={OUTPUTSIZE}&apikey={API_KEY}&datatype=csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("Raw Response First 500 chars:")
        print(response.text[:500])
        
        # Check if response contains error or note usually returned in CSV format as text?
        # AlphaVantage CSV usually comes clean, but if error it might be json inside csv?
        # Let's peek at the first few bytes
        content_sample = response.content[:100].decode('utf-8')
        if "Error Message" in content_sample or "Note" in content_sample:
            print("API Warning/Error:", content_sample)
            if "Error Message" in content_sample:
                return None
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(response.content))
        print(f"Fetched {len(df)} rows from API.")
        print(f"Columns found: {df.columns.tolist()}")
        print(f"Head:\n{df.head()}")
        
        # Rename columns to match our standard
        # AlphaVantage CSV: timestamp,open,high,low,close
        df.rename(columns={
            'timestamp': 'Date', 
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close'
        }, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def load_local_data():
    print("Loading local data...")
    if not os.path.exists(EXISTING_DATA_PATH):
        print("Local file not found!")
        return None
        
    df = pd.read_csv(EXISTING_DATA_PATH)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse timestamps - handling the specific format "01.07.2025 23:59:00.000 GMT+0530" if present, 
    # or relying on vector_backtest's logic
    # Note: vector_backtest uses: pd.to_datetime(df['timestamp'], dayfirst=True, utc=True, errors='coerce')
    
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True, errors='coerce')
    else:
        # Fallback if headers are different, though we saw 'timestamp' before
        print("Warning: 'timestamp' column not found, attempting auto-detection")
        df['Date'] = pd.to_datetime(df.iloc[:,0], dayfirst=True, utc=True, errors='coerce')

    df.set_index('Date', inplace=True)
    df.index = df.index.tz_convert(None) # Remove TZ for merging consistency maybe? Or keep UTC? 
    # AlphaVantage is usually UTC. Local data had GMT+0530 but we converted to UTC.
    # Let's verify: df['Date'] was converted to UTC.
    # To be safe, localize everything to None (Naive) but treated as UTC.
    
    # NOTE: pd.to_datetime(..., utc=True) makes it TZ-aware (UTC).
    # If we want to align, we should keep both TZ-aware or both TZ-naive.
    # AlphaVantage above: pd.to_datetime(..., utc=True)
    # So both are UTC. Good.
    
    # Select cols
    cols_map = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}
    # Handle casing in local file
    cols_found = {}
    for c in df.columns:
        if c.capitalize() in cols_map:
            cols_found[c] = cols_map[c.capitalize()]
    
    df = df.rename(columns=cols_found)
    return df[['Open', 'High', 'Low', 'Close']]

def main():
    new_df = fetch_alphavantage_data()
    
    if new_df is not None:
        print(f"Fetched {len(new_df)} daily records.")
        # Filter for recent years if needed, or keep all
        # new_df = new_df[new_df.index.year >= 2025]
        
        new_df.to_csv(OUTPUT_FILENAME)
        print(f"Saved Daily data to {OUTPUT_FILENAME}")
        print("Recent Data (Head):")
        print(new_df.head())
        print("Recent Data (Tail):")
        print(new_df.tail())
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()
