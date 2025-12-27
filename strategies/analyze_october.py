import pandas as pd
from engine import db_client

# 1. Fetch Data from DB
print("Fetching October 2025 data from DB...")
# We fetch a bit more to handle indicators if needed, but for simple analysis specific range is fine
try:
    oct_data = db_client.get_market_data(start_date='2025-10-01', end_date='2025-10-31')
except Exception as e:
    print(f"Error fetching data: {e}")
    exit(1)

print(f"October 2025 Data Points: {len(oct_data)}")

if len(oct_data) == 0:
    print("No data for October 2025.")
else:
    # Resample to daily if the data is minute-level (typical for this DB)
    # Check resolution
    time_diff = oct_data.index[1] - oct_data.index[0]
    if time_diff.total_seconds() < 86400:
        print("Resampling Minute data to Daily...")
        daily_agg = {
            'Open': 'first', 
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum'
        }
        oct_data = oct_data.resample('D').agg(daily_agg).dropna()

    start_price = oct_data['Open'].iloc[0]
    end_price = oct_data['Close'].iloc[-1]
    change = end_price - start_price
    pct_change = (change / start_price) * 100
    
    high = oct_data['High'].max()
    low = oct_data['Low'].min()
    range_pts = high - low
    
    avg_daily_range = (oct_data['High'] - oct_data['Low']).mean()
    
    print(f"Start Price: {start_price}")
    print(f"End Price: {end_price}")
    print(f"Change: {change:.3f} ({pct_change:.2f}%)")
    print(f"High: {high} | Low: {low} | Range: {range_pts:.3f}")
    print(f"Avg Daily Range: {avg_daily_range:.3f}")
    
    # Check for consecutive green/red days
    oct_data['Direction'] = oct_data['Close'] > oct_data['Open']
    print("\nDaily Direction (True=Green, False=Red):")
    print(oct_data['Direction'].value_counts())
    
    print("\nDaily Data:")
    print(oct_data[['Open', 'High', 'Low', 'Close']])
