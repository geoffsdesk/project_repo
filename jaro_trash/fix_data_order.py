import pandas as pd

file_path = "data/USD_JPY_DAILY_FIXED.csv"
df = pd.read_csv(file_path)

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort
df.sort_values('Date', inplace=True)

# Drop duplicates
df.drop_duplicates(subset=['Date'], inplace=True)

# Save back
df.to_csv(file_path, index=False)
print("Sorted and cleaned CSV.")
