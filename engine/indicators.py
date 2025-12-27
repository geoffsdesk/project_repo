import pandas as pd
import numpy as np

def calculate_value_areas(df: pd.DataFrame, va_percent=0.70) -> pd.DataFrame:
    """
    Calculates Daily Value Area (VAH, VAL, POC) and maps them to the timeframe.
    Assumes df has DateTimeIndex and 'High', 'Low', 'Close', 'Open'.
    """
    # Resample to Daily
    daily = df.resample('D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    
    # Calculate VA Proxy
    daily['POC'] = (daily['High'] + daily['Low'] + daily['Close']) / 3
    daily['Range'] = daily['High'] - daily['Low']
    daily['VAH'] = daily['POC'] + (daily['Range'] * va_percent / 2)
    daily['VAL'] = daily['POC'] - (daily['Range'] * va_percent / 2)
    
    # Shift to get "Previous Day's" values
    daily['PrevVAH'] = daily['VAH'].shift(1)
    daily['PrevVAL'] = daily['VAL'].shift(1)
    daily['PrevPOC'] = daily['POC'].shift(1)
    
    # Map back to original DF
    df['DateOnly'] = df.index.date
    daily['DateOnly'] = daily.index.date
    daily_indexed = daily.set_index('DateOnly')
    
    # Map
    df['PrevVAH'] = df['DateOnly'].map(daily_indexed['PrevVAH'])
    df['PrevVAL'] = df['DateOnly'].map(daily_indexed['PrevVAL'])
    df['PrevPOC'] = df['DateOnly'].map(daily_indexed['PrevPOC'])
    
    # Today Open (for logic that needs "Open of the day")
    day_open_serie = daily.set_index('DateOnly')['Open']
    df['TodayOpen'] = df['DateOnly'].map(day_open_serie)
    
    # Clean up
    df.drop(columns=['DateOnly'], inplace=True, errors='ignore')
    return df

def calculate_acceptance(df: pd.DataFrame, period='30min') -> pd.DataFrame:
    """
    Calculates 80% Rule Acceptance:
    Two consecutive closes of 'period' timeframe inside the previous day's VA.
    """
    # We need to operate on the resampled data but apply checks against PrevVAH/VAL
    # Check 1: Resample closes
    resampled = df.resample(period).agg({'Close': 'last'})
    resampled['DateOnly'] = resampled.index.date
    
    # We need the VAH/VAL map on 30m too. Re-use existing map if possible, but simplest is to map again.
    # To avoid circular dependency, we assume 'PrevVAH' isn't on df yet or we re-derive.
    # Actually, let's assume df ALREADY has PrevVAH/PrevVAL (from calc_value_areas).
    # We can take the first value of PrevVAH for that period.
    
    # Helper to get the VAH/VAL for the resampled blocks
    # We can just re-map using the date of the resampled block
    # But getting the daily data again is inefficient. 
    # Let's assume we can get it from the 'df' by downsampling 'PrevVAH'.
    
    # Resample VAH/VAL (taking 'first' is safe as they are constant for the day)
    res_va = df.resample(period).agg({'PrevVAH': 'first', 'PrevVAL': 'first'})
    
    resampled = resampled.join(res_va)
    
    # Check Inside
    resampled['IsInside'] = (resampled['Close'] < resampled['PrevVAH']) & (resampled['Close'] > resampled['PrevVAL'])
    
    # Check 2 Consecutive
    resampled['Accepted'] = resampled['IsInside'].shift(1) & resampled['IsInside'].shift(2)
    
    # Map back to DF
    # We map based on the 'floor' time
    df['TimeBlock'] = df.index.floor(period)
    df['Accepted'] = df['TimeBlock'].map(resampled['Accepted']).fillna(False).astype(bool)
    
    df.drop(columns=['TimeBlock'], inplace=True, errors='ignore')
    return df
