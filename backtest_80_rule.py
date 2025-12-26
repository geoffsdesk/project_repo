import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Configuration ---
# --- Configuration ---
DATA_PATH = r"C:\Users\geoff\.gemini\antigravity\scratch\project_repo\USDJPY 1M_CANDLESTICK DATA 2015-2025\USD_JPY_2015_07_2025_BID.csv"
VA_PERCENT = 0.70
LOOKBACK = 1
INITIAL_CAPITAL = 10000
POSITION_SIZE = 1000  # Fixed 1000 units (Micro Lot)
SL_TICKS = 50
TP_TICKS = 100 
USE_OPPOSITE_VA_TP = True

class ValueAreaStrategy(Strategy):
    sl_ticks = SL_TICKS
    tp_ticks = TP_TICKS
    
    def init(self):
        pass

    def next(self):
        current_time = self.data.index[-1]
        price = self.data.Close[-1]
        day_open = self.data.TodayOpen[-1]
        
        prev_vah = self.data.PrevVAH[-1]
        prev_val = self.data.PrevVAL[-1]
        
        open_above = day_open > prev_vah
        open_below = day_open < prev_val
        
        accepted = self.data.Accepted[-1]
        
        # Entry Logic
        if not self.position:
            if accepted and open_above:
                # SHORT
                sl_price = price + self.sl_ticks * 0.001 
                tp_price = prev_val if USE_OPPOSITE_VA_TP else price - self.tp_ticks * 0.001
                self.sell(sl=sl_price, tp=tp_price, size=POSITION_SIZE)
                
            elif accepted and open_below:
                # LONG
                sl_price = price - self.sl_ticks * 0.001
                tp_price = prev_vah if USE_OPPOSITE_VA_TP else price + self.tp_ticks * 0.001
                self.buy(sl=sl_price, tp=tp_price, size=POSITION_SIZE)
# ... [Keeping preprocess_data as is] ...

# ... [Inside if __name__ == "__main__":] ...
        # ...
        if len(subset) > 0:
            # Add Margin (1:50 leverage = 0.02)
            bt = Backtest(subset, ValueAreaStrategy, cash=INITIAL_CAPITAL, commission=.0002, margin=0.02)
            stats = bt.run()
            print(stats)
            
            # Print trade list head
            # print(stats['_trades'].head())
        else:
            print("Skipping backtest due to empty data.")

def preprocess_data(filepath):
    print("Loading data...")
    # Load without index first to inspect/convert
    df = pd.read_csv(filepath)
    
    # Clean whitespace in column names
    df.columns = df.columns.str.strip()
    
    # Parse dates explicitly
    # Format sample: 01.01.2015 00:00:00.000 GMT+0530
    # Pandas to_datetime can handle this usually if we give it a hint or let it guess with dayfirst=True
    # The timezone might be tricky. Let's try flexible parsing or specific format if needed.
    # Note: 'mixed' format or 'ISO8601' not strictly matching this.
    
    print("Parsing timestamps (this may take a moment)...")
    # Handling "01.01.2015 00:00:00.000 GMT+0530"
    # We strip timezone for simplicity if needed, or convert to UTC.
    
    # Helper to strip ' GMT.*' if simple parsing fails, but let's try standard first.
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True)
    except Exception as e:
        print(f"Standard parsing failed: {e}. Trying slower custom parser or format...")
        # Fallback or more specific format
        # Trying to ignore the GMT part if it varies or standard parser dislikes it
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True, utc=True)
        
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None) # Strip timezone to avoid recursion error
    df.index.name = 'Date'
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Clean output
    df = df[['Open', 'High', 'Low', 'Close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Sort index just in case
    df.sort_index(inplace=True)
    
    # Ensure numeric types
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df.dropna(inplace=True)
    
    # --- 1. Resample to Daily for VA Calculation ---
    print("Resampling Daily...")
    daily_df = df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # Calculate VA Proxy
    # POC = (H+L+C)/3
    # Range = H-L
    # VAH = POC + (Range * 0.7 / 2)
    daily_df['POC'] = (daily_df['High'] + daily_df['Low'] + daily_df['Close']) / 3
    daily_df['Range'] = daily_df['High'] - daily_df['Low']
    daily_df['VAH'] = daily_df['POC'] + (daily_df['Range'] * VA_PERCENT / 2)
    daily_df['VAL'] = daily_df['POC'] - (daily_df['Range'] * VA_PERCENT / 2)
    
    # Shifts: We need YESTERDAY'S values for TODAY
    daily_df['PrevVAH'] = daily_df['VAH'].shift(1)
    daily_df['PrevVAL'] = daily_df['VAL'].shift(1)
    daily_df['PrevPOC'] = daily_df['POC'].shift(1)
    
    # Merge Daily levels back to 1-minute data
    # ffill ensures every minute of today gets yesterday's VA values
    print("Merging Daily Levels...")
    df['DateOnly'] = df.index.date
    daily_df['DateOnly'] = daily_df.index.date
    
    merged = df.merge(daily_df[['DateOnly', 'PrevVAH', 'PrevVAL', 'PrevPOC']], on='DateOnly', how='left')
    merged.index = df.index # Restore DatetimeIndex
    
    # Add 'TodayOpen' (Open price of the current day)
    # We can group by DateOnly and transform 'first'
    merged['TodayOpen'] = merged.groupby('DateOnly')['Open'].transform('first')
    
    # --- 2. Acceptance Logic (30m) ---
    print("Calculating Acceptance...")
    # Resample 1m to 30m to check "Inside" logic
    res_30m = df.resample('30min').agg({'Close': 'last'})
    
    # Map VA levels to 30m bars
    # (Doing this alignment is tricky, easier to do logic on 1m if we know "current 30m is inside")
    # Actually, the requirement is "2 consecutive 30m bars inside".
    
    # Let's map 30m closes back to 1m
    # Each 1m bar needs to know: "Did the LAST COMPLETED 30m bar close inside?"
    # and "Did the one BEFORE THAT close inside?"
    
    # We can reindex 30m closes to 1m (ffill)
    # But strictly speaking, at 10:15, the "last 30m bar" is the 10:00 candle? No, it's the 09:30-10:00 candle.
    # The current 30m candle (10:00-10:30) is not closed.
    
    res_30m['Close30m'] = res_30m['Close']
    
    # Merge 30m data
    # We use 'asof' or simply reindex with method='ffill' BUT we must lag it by 1 period
    # because at 10:00:00 we know the close of 09:30:00 bar.
    # Actually, simpler:
    # 1. Calculate 'IsInside' on the 30m dataframe using the Daily VA levels (we need to merge daily to 30m first)
    
    res_30m['DateOnly'] = res_30m.index.date
    res_30m = res_30m.merge(daily_df[['DateOnly', 'PrevVAH', 'PrevVAL']], on='DateOnly', how='left')
    res_30m.index = pd.date_range(start=res_30m['DateOnly'].iloc[0], periods=len(res_30m), freq='30min') # Fix index lost by merge if needed? 
    # Warning: Merge destroys index usually. Let's be careful.
    
    # Better approach for 30m:
    # Index align daily to 30m
    # Check IsInside
    
    # Let's stick to the 1m dataframe 'merged' and use 'resample().last()' smartly.
    
    merged['Time30m'] = merged.index.floor('30min')
    
    # Group by 30m blocks to get Close of that block
    # But we need PAST 30m blocks.
    grp_30m = merged.groupby('Time30m')['Close'].last()
    
    # Check if that close was inside VA (we need VA of that time)
    # We can get VA from 'merged' by taking the first value of that group
    vah_30m = merged.groupby('Time30m')['PrevVAH'].first()
    val_30m = merged.groupby('Time30m')['PrevVAL'].first()
    
    is_inside_30m = (grp_30m < vah_30m) & (grp_30m > val_30m)
    
    # We need 2 consecutive closes
    # shift(1) here means "Previous 30m bar"
    # shift(2) means "2 bars ago"
    # At any time T (inside a 30m bar), we want to know if T-1_30m and T-2_30m were inside.
    
    accepted_30m = is_inside_30m.shift(1) & is_inside_30m.shift(2)
    
    # Now map 'accepted_30m' back to 1m
    # We can use map based on 'Time30m' column
    # Note: accepted_30m index is Timestamp of the interval start.
    merged['Accepted'] = merged['Time30m'].map(accepted_30m)
    
    # Fill NAs
    merged.dropna(inplace=True)
    
    # Drop temp columns
    cols_to_drop = ['DateOnly', 'Time30m', 'Close30m'] 
    merged.drop(columns=[c for c in cols_to_drop if c in merged.columns], inplace=True)
    
    # Cast boolean to int
    if 'Accepted' in merged.columns:
        merged['Accepted'] = merged['Accepted'].astype(int)
        
    # Final type check debug
    print("Final Types:")
    print(merged.dtypes)
    
    return merged


if __name__ == "__main__":
    # Redirect print to file for reliable inspection
    import sys
    # Use standard open, assuming CWD is correct
    with open("debug_output.txt", "w") as f:
        sys.stdout = f
        
        try:
            data = preprocess_data(DATA_PATH)
            
            # Filter for User Date Range (Jan 2025 only for test)
            start_date = "2025-01-01"
            end_date = "2025-02-01"
            mask = (data.index >= start_date) & (data.index <= end_date)
            subset = data.loc[mask]
            
            print(f"Data Date Range: {data.index.min()} to {data.index.max()}")
            print(f"Subset Date Range: {subset.index.min()} to {subset.index.max()}")
            print(f"Subset Rows: {len(subset)}")
            
            if len(subset) == 0:
                print("Error: Subset is empty. Check date parsing or range.")
            else:
                print("Checking Signals in Subset:")
                print(f"Accepted Count: {subset['Accepted'].sum()}")
                try:
                    print(f"PrevVAH Sample: {subset['PrevVAH'].dropna().iloc[0]}")
                except:
                     print("PrevVAH all NaNs")
                try:
                    print(f"TodayOpen Sample: {subset['TodayOpen'].dropna().iloc[0]}")
                except:
                     print("TodayOpen all NaNs")
                
            print(f"Running backtest on {len(subset)} rows...")
            
            if len(subset) > 0:
                bt = Backtest(subset, ValueAreaStrategy, cash=INITIAL_CAPITAL, commission=.0002, margin=0.02)
                stats = bt.run()
                print(stats)
            else:
                print("Skipping backtest due to empty data.")
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            
    sys.stdout = sys.__stdout__
