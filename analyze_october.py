import pandas as pd

# Load the daily data
df = pd.read_csv("USD_JPY_DAILY_AUGMENTED.csv", index_col='Date', parse_dates=True)
df.sort_index(inplace=True) # Ensure monotonic increasing for slicing

# Filter for October 2025
oct_data = df['2025-10-01':'2025-10-31'].sort_index()

print(f"October 2025 Data Points: {len(oct_data)}")

if len(oct_data) == 0:
    print("No data for October 2025.")
else:
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
    
    # Check for strong trend days (Close near High/Low)
    # ...
    
    print("\nDaily Data:")
    print(oct_data[['Open', 'High', 'Low', 'Close']])
