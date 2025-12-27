import pandas as pd
import sys

input_file = "data/USD_JPY_DAILY_AUGMENTED.csv"
output_file = "data/USD_JPY_DAILY_FIXED.csv"

try:
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Check if Volume exists
    if 'Volume' not in df.columns:
        print("Adding dummy Volume column...")
        df['Volume'] = 1000 # Dummy volume
        df.to_csv(output_file, index=False)
        print(f"Saved fixed CSV to {output_file}")
    else:
        print("Volume column already exists.")
        
except Exception as e:
    print(f"Error: {e}")
