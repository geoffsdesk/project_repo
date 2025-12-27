import pandas as pd
from engine import db_client

def export_data():
    start_date = '2025-01-01'
    end_date = '2025-06-30'
    output_file = 'data/USD_JPY_2025_H1.csv'
    
    print(f"Fetching data from {start_date} to {end_date}...")
    try:
        df = db_client.get_market_data(start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print("No data found for this range.")
        return

    print(f"Fetched {len(df)} rows.")
    
    # Ensure format for PyneCore: Date, Open, High, Low, Close, Volume
    # db_client returns 'timestamp' as index.
    df.reset_index(inplace=True)
    
    # Rename 'timestamp' to 'Date'
    df.rename(columns={'timestamp': 'Date', 'Volume': 'Volume'}, inplace=True)
    
    # Ensure Volume exists
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    else:
        df['Volume'] = df['Volume'].fillna(0)

    # Upper case columns just in case
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
    }, inplace=True)
    
    # Select and order columns
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # Check if we have them all
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        # available:
        print(f"Available: {df.columns.tolist()}")
        return

    df = df[cols]
    
    # Sort
    df.sort_values('Date', inplace=True)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"Successfully exported to {output_file}")

if __name__ == "__main__":
    export_data()
