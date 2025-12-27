import pandas as pd
import numpy as np
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import argparse

# --- Configuration ---
DATA_PATH = r"C:\Users\geoff\.gemini\antigravity\scratch\project_repo\USDJPY 1M_CANDLESTICK DATA 2015-2025\USD_JPY_2015_07_2025_BID.csv"
INITIAL_CAPITAL = 10000
POSITION_SIZE = 1000  # Units
LEVERAGE = 50
SL_TICKS = 50
TP_TICKS = 100 
USE_OPPOSITE_VA_TP = True

def preprocess_data(filepath, va_percent=0.70):
    print(f"Loading data with VA Percent: {va_percent}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # Fast Parse
    # Use coerce to handle any errors, assuming mostly valid
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True, errors='coerce')
    df.set_index('timestamp', inplace=True)
    
    # We will keep the index as UTC for now, but for Jaro we might need NY time conversion
    # df.index = df.index.tz_localize(None) 
    # ^ Removing this to keep it aware, or we can strip after conversion. 
    # Let's strip to be consistent with previous logic but remember it's UTC.
    df.index = df.index.tz_localize(None)
    
    df.index.name = 'Date'
    df = df[~df.index.duplicated(keep='first')]
    
    # Cast
    cols = ['Open', 'High', 'Low', 'Close', 'volume']
    # Handle casing if 'volume' is lowercase in CSV
    df_cols_map = {c: c for c in df.columns}
    if 'volume' in df_cols_map and 'Volume' not in df_cols_map:
        df.rename(columns={'volume': 'Volume'}, inplace=True)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # Daily VA
    print("Calculating VA...")
    daily = df.resample('D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    daily['POC'] = (daily['High'] + daily['Low'] + daily['Close'])/3
    daily['Range'] = daily['High'] - daily['Low']
    daily['VAH'] = daily['POC'] + (daily['Range']*va_percent/2)
    daily['VAL'] = daily['POC'] - (daily['Range']*va_percent/2)
    daily['PrevVAH'] = daily['VAH'].shift(1)
    daily['PrevVAL'] = daily['VAL'].shift(1)
    daily['DateOnly'] = daily.index.date
    
    # Merge/Map Daily Levels
    # Use map to preserve index (merge resets it)
    print("Mapping Daily Levels...")
    df['DateOnly'] = df.index.date
    
    daily_indexed_by_date = daily.set_index('DateOnly')
    
    # Map VAH/VAL/PrevVAH etc if needed
    # We need PrevVAH (Yesterday's value)
    df['PrevVAH'] = df['DateOnly'].map(daily_indexed_by_date['PrevVAH'])
    df['PrevVAL'] = df['DateOnly'].map(daily_indexed_by_date['PrevVAL'])
    
    # Today Open
    # df['TodayOpen'] = df.groupby('DateOnly')['Open'].transform('first') 
    # Transform is slow.
    day_open_map = daily.set_index('DateOnly')['Open']
    df['TodayOpen'] = df['DateOnly'].map(day_open_map)
    
    # Acceptance (30m)
    print("Calculating Acceptance...")
    # Resample to 30m
    res30 = df.resample('30min').agg({'Close':'last'})
    # We need the VA levels for these 30m bars.
    # Map from Daily
    res30['DateOnly'] = res30.index.date
    res30['PrevVAH'] = res30['DateOnly'].map(daily_indexed_by_date['PrevVAH'])
    res30['PrevVAL'] = res30['DateOnly'].map(daily_indexed_by_date['PrevVAL'])
    
    # Is Inside
    res30['IsInside'] = (res30['Close'] < res30['PrevVAH']) & (res30['Close'] > res30['PrevVAL'])
    # Accepted = 2 consecutive previous inside
    res30['Accepted'] = res30['IsInside'].shift(1) & res30['IsInside'].shift(2)
    
    # Map back to 1m
    df['Time30m'] = df.index.floor('30min')
    df['Accepted'] = df['Time30m'].map(res30['Accepted']).fillna(False).astype(bool)
    
    # Clean
    df.dropna(subset=['PrevVAH', 'TodayOpen'], inplace=True)
    return df

def run_strategy_80_rule(df):
    print("Running Strategy: 80% Rule...")
    status = 'flat'
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    
    trades = []
    equity = INITIAL_CAPITAL
    
    # Arrays for speed
    highs = df.High.values
    lows = df.Low.values
    closes = df.Close.values
    times = df.index
    
    prev_vah = df.PrevVAH.values
    prev_val = df.PrevVAL.values
    today_open = df.TodayOpen.values
    accepted = df.Accepted.values
    
    n = len(df)
    entry_time = None 
    
    for i in range(n):
        price = closes[i]
        
        # Check Exits first
        if status == 'long':
            # Check Low for SL, High for TP
            if lows[i] <= sl:
                # SL Hit
                exit_price = sl 
                pnl = (exit_price - entry_price) * POSITION_SIZE
                equity += pnl
                trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'SL'})
                status = 'flat'
            elif highs[i] >= tp:
                # TP Hit
                exit_price = tp
                pnl = (exit_price - entry_price) * POSITION_SIZE
                equity += pnl
                trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'TP'})
                status = 'flat'
                
        elif status == 'short':
             if highs[i] >= sl:
                # SL Hit
                exit_price = sl
                pnl = (entry_price - exit_price) * POSITION_SIZE
                equity += pnl
                trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Short', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'SL'})
                status = 'flat'
             elif lows[i] <= tp:
                # TP Hit
                exit_price = tp
                pnl = (entry_price - exit_price) * POSITION_SIZE
                equity += pnl
                trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Short', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'TP'})
                status = 'flat'
        
        # Check Entry
        if status == 'flat':
            if accepted[i]:
                is_above = today_open[i] > prev_vah[i]
                is_below = today_open[i] < prev_val[i]
                
                if is_above:
                    # Short
                    entry_price = price
                    entry_time = times[i]
                    sl = price + SL_TICKS * 0.001
                    tp = prev_val[i] if USE_OPPOSITE_VA_TP else price - TP_TICKS * 0.001
                    status = 'short'
                elif is_below:
                    # Long
                    entry_price = price
                    entry_time = times[i]
                    sl = price - SL_TICKS * 0.001
                    tp = prev_vah[i] if USE_OPPOSITE_VA_TP else price + TP_TICKS * 0.001
                    status = 'long'
                    
    return equity, trades, "80% Rule"

def run_strategy_jaro_v1(df):
    print("Running Strategy: Jaro V1...")
    
    # --- Jaro V1 Logic ---
    # VA Percent is 45% (Handled in preprocess)
    
    # Time Conversion for Logic
    # Data is UTC. New York is what we need.
    # We need to construct a NY time array for filtering.
    # To avoid heavy lifting inside the loop, we pre-calculate time of day minutes in NY.
    
    # Assuming df.index is Naive UTC.
    # Localize to UTC then Convert to NY
    ts_utc = df.index.tz_localize('UTC')
    ts_ny = ts_utc.tz_convert('America/New_York')
    
    # Calculate Minutes from Midnight for each bar
    minutes_from_midnight = ts_ny.hour * 60 + ts_ny.minute
    minutes_from_midnight = minutes_from_midnight.values # Numpy array
    
    # Logic Constants
    TIME_ENTRY_Limit = 900  # 15:00 (3:00 PM)
    TIME_EOD_FLUSH = 960    # 16:00 (4:00 PM)
    TIME_HARD_CLOSE = 1005  # 16:45 (4:45 PM)
    
    # State Variables
    status = 'flat' # 'flat', 'long', 'short'
    trades_today = 0
    tp_candle_level = np.nan
    
    entry_price = 0.0
    sl = 0.0
    tp = 0.0 # Exit limit
    
    trades = []
    equity = INITIAL_CAPITAL
    
    # Pre-calculate Day changes to reset counters
    # shift(1) of date != date
    dates = df.index.date
    # Day change detection
    day_indices = np.where(dates[1:] != dates[:-1])[0] + 1
    day_change_mask = np.zeros(len(df), dtype=bool)
    day_change_mask[day_indices] = True
    
    # Arrays
    opens = df.Open.values
    highs = df.High.values
    lows = df.Low.values
    closes = df.Close.values
    times = df.index
    
    prev_vah = df.PrevVAH.values
    prev_val = df.PrevVAL.values
    today_open = df.TodayOpen.values
    accepted = df.Accepted.values
    
    # Calculate VA Spread for Targets
    va_spread = prev_vah - prev_val
    long_target_t1 = prev_val + (va_spread * 0.75)
    short_target_t1 = prev_vah - (va_spread * 0.75)
    
    n = len(df)
    entry_time = None
    
    for i in range(n):
        # 1. New Day Reset
        if day_change_mask[i]:
            trades_today = 0
            tp_candle_level = np.nan
            if status != 'flat':
                # Force close at open of new day if still open? 
                # Strategy says EOD Close at 16:45. 
                # If we missed it, close now.
                pass 
                
        current_minutes = minutes_from_midnight[i]
        price = closes[i]
        
        # 2. Exits & EOD
        if status != 'flat':
            # EOD Logic
            if current_minutes >= TIME_HARD_CLOSE:
                # Force Close
                 pnl = (price - entry_price) * POSITION_SIZE if status == 'long' else (entry_price - price) * POSITION_SIZE
                 equity += pnl
                 trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': status.capitalize(), 'Entry': entry_price, 'Exit': price, 'PnL': pnl, 'Reason': 'EOD Risk Cut'})
                 status = 'flat'
                 
            elif current_minutes >= TIME_EOD_FLUSH:
                # Check for open profit
                open_pnl = (price - entry_price) * POSITION_SIZE if status == 'long' else (entry_price - price) * POSITION_SIZE
                if open_pnl > 0:
                    equity += open_pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': status.capitalize(), 'Entry': entry_price, 'Exit': price, 'PnL': open_pnl, 'Reason': 'EOD Profit Flush'})
                    status = 'flat'
            
            # Normal TP/SL Logic (if still open)
            if status == 'long':
                if lows[i] <= sl:
                    exit_price = sl
                    pnl = (exit_price - entry_price) * POSITION_SIZE
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'SL'})
                    status = 'flat'
                elif highs[i] >= tp:
                    exit_price = tp
                    pnl = (exit_price - entry_price) * POSITION_SIZE
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'TP'})
                    status = 'flat'
                    # Capture Momentum Level
                    if trades_today == 1:
                        tp_candle_level = highs[i] # Trade 1 Long TP -> High

            elif status == 'short':
                if highs[i] >= sl:
                    exit_price = sl
                    pnl = (entry_price - exit_price) * POSITION_SIZE
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Short', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'SL'})
                    status = 'flat'
                elif lows[i] <= tp:
                    exit_price = tp
                    pnl = (entry_price - exit_price) * POSITION_SIZE
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Short', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'TP'})
                    status = 'flat'
                    # Capture Momentum Level
                    if trades_today == 1:
                        tp_candle_level = lows[i] # Trade 1 Short TP -> Low
                        
        # 3. Entries
        can_enter = current_minutes < TIME_ENTRY_Limit
        
        if status == 'flat' and can_enter:
            
            # TRADE 1: Initial
            if trades_today == 0 and accepted[i]:
                # Short
                if today_open[i] > prev_vah[i]:
                    status = 'short'
                    entry_price = price
                    entry_time = times[i]
                    # SL/TP
                    sl = prev_vah[i] + SL_TICKS * 0.001 # Strategy: stop=prevVAH + 50 ticks (approx 0.050 or 0.50? JPY 0.01 is pip. 50*0.001 = 0.05)
                    # wait, syminfo.mintick * 50. JPY mintick is 0.001. So 50 * 0.001 = 0.05.
                    tp = short_target_t1[i]
                    trades_today = 1
                
                # Long
                elif today_open[i] < prev_val[i]:
                    status = 'long'
                    entry_price = price
                    entry_time = times[i]
                    sl = prev_val[i] - SL_TICKS * 0.001
                    tp = long_target_t1[i]
                    trades_today = 1
                    
            # TRADE 2: Re-entry
            elif trades_today == 1 and not np.isnan(tp_candle_level):
                # Long Re-entry
                # If price breaks ABOVE the TP candle high (tp_candle_level)
                # And todayOpen < prevVAL (Context)
                if today_open[i] < prev_val[i] and price > tp_candle_level:
                     status = 'long'
                     entry_price = price
                     entry_time = times[i]
                     sl = prev_val[i] - SL_TICKS * 0.001
                     tp = prev_vah[i] # Target: Opposite VA
                     trades_today = 2
                
                # Short Re-entry
                # If price breaks BELOW the TP candle low
                # And todayOpen > prevVAH
                elif today_open[i] > prev_vah[i] and price < tp_candle_level:
                    status = 'short'
                    entry_price = price
                    entry_time = times[i]
                    sl = prev_vah[i] + SL_TICKS * 0.001
                    tp = prev_val[i] # Target: Opposite VA
                    trades_today = 2

    return equity, trades, "Jaro V1"

def plot_results(df, trades, strategy_name):
    print(f"Generating Chart for {strategy_name}...")
    
    # --- Metrics Calculation ---
    df_trades = pd.DataFrame(trades)
    
    if len(trades) > 0:
        df_trades.sort_values('ExitTime', inplace=True)
        df_trades['CumulativePnL'] = df_trades['PnL'].cumsum()
        df_trades['Equity'] = INITIAL_CAPITAL + df_trades['CumulativePnL']
        
        # Drawdown
        running_peak = df_trades['Equity'].cummax()
        drawdown = running_peak - df_trades['Equity']
        drawdown_pct = (drawdown / running_peak) * 100
        max_dd_pct = drawdown_pct.max()
        
        # Stats
        total_pnl = df_trades['PnL'].sum()
        wins = df_trades[df_trades['PnL'] > 0]
        losses = df_trades[df_trades['PnL'] <= 0]
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        win_rate = (len(wins) / len(df_trades)) * 100
        total_trades = len(df_trades)
        
        eq_dates = [df.index[0]] + df_trades['ExitTime'].tolist()
        eq_values = [INITIAL_CAPITAL] + df_trades['Equity'].tolist()
    else:
        # Defaults
        total_pnl = 0
        max_dd_pct = 0
        profit_factor = 0
        win_rate = 0
        total_trades = 0
        eq_dates = [df.index[0]]
        eq_values = [INITIAL_CAPITAL]

    # --- Plotting ---
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.5, 0.2, 0.3],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=(f'Price Action ({strategy_name})', 'Equity Curve', 'Trade List')
    )
    
    # 1. Price Candle
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # 2. Indicators (VAH, VAL)
    fig.add_trace(go.Scatter(x=df.index, y=df['PrevVAH'], mode='lines', line=dict(color='green', width=1), name='VAH'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['PrevVAL'], mode='lines', line=dict(color='red', width=1), name='VAL'), row=1, col=1)
    
    # 3. Trades Markers
    if len(trades) > 0:
        long_entries_t = [t['EntryTime'] for t in trades if t['Type'] == 'Long']
        long_entries_p = [t['Entry'] for t in trades if t['Type'] == 'Long']
        
        short_entries_t = [t['EntryTime'] for t in trades if t['Type'] == 'Short']
        short_entries_p = [t['Entry'] for t in trades if t['Type'] == 'Short']
        
        exits_t = [t['ExitTime'] for t in trades]
        exits_p = [t['Exit'] for t in trades]
        pnl_text = [f"PnL: {t['PnL']:.2f}<br>{t['Reason']}" for t in trades]
        
        if long_entries_t:
            fig.add_trace(go.Scatter(
                x=long_entries_t, y=long_entries_p,
                mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
                name='Long Entry'
            ), row=1, col=1)
            
        if short_entries_t:
            fig.add_trace(go.Scatter(
                x=short_entries_t, y=short_entries_p,
                mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
                name='Short Entry'
            ), row=1, col=1)
            
        if exits_t:
            fig.add_trace(go.Scatter(
                x=exits_t, y=exits_p,
                mode='markers', marker=dict(symbol='x', color='blue', size=8),
                text=pnl_text,
                name='Exit'
            ), row=1, col=1)

    # 4. Equity Curve
    fig.add_trace(go.Scatter(
        x=eq_dates, y=eq_values,
        mode='lines', 
        name='Equity',
        line=dict(color='cyan', width=2),
        fill='tozeroy'
    ), row=2, col=1)
    
    # 5. Trade Table
    if len(trades) > 0:
        t_entry_time = [t['EntryTime'].strftime('%Y-%m-%d %H:%M') for t in trades]
        t_type = [t['Type'] for t in trades]
        t_entry_p = [f"{t['Entry']:.3f}" for t in trades]
        t_reason = [t['Reason'] for t in trades]
        t_exit_time = [t['ExitTime'].strftime('%Y-%m-%d %H:%M') for t in trades]
        t_exit_p = [f"{t['Exit']:.3f}" for t in trades]
        t_pnl = [f"{t['PnL']:.2f}" for t in trades]
        
        fill_colors = ['palegreen' if t['PnL'] > 0 else 'lightpink' for t in trades]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['Entry Time', 'Type', 'Reason', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL'],
                fill_color='grey',
                align='left',
                font=dict(color='white')
            ),
            cells=dict(
                values=[t_entry_time, t_type, t_reason, t_entry_p, t_exit_time, t_exit_p, t_pnl],
                fill_color=[['black']*len(trades), ['black']*len(trades), ['black']*len(trades), ['black']*len(trades), ['black']*len(trades), ['black']*len(trades), fill_colors],
                align='left',
                font=dict(color=['white', 'white', 'white', 'white', 'white', 'white', 'black'])
            )
        ), row=3, col=1)

    # Layout & Title
    stats_text = (
        f"<b>Total P&L:</b> {total_pnl:.2f} | "
        f"<b>Max Drawdown:</b> {max_dd_pct:.2f}% | "
        f"<b>Profit Factor:</b> {profit_factor:.2f} | "
        f"<b>Win Rate:</b> {win_rate:.1f}% ({len(wins)}/{total_trades})"
    )
    
    fig.update_layout(
        title=dict(text=f'Backtest Results: {strategy_name}<br><sup>{stats_text}</sup>', x=0.5),
        yaxis_title='USD/JPY',
        yaxis2_title='Equity ($)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=1000
    )
    
    filename = f"backtest_chart_{strategy_name.replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"Chart saved to {filename}")
    webbrowser.open('file://' + os.path.realpath(filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Vector Backtest')
    parser.add_argument('--strategy', type=str, default='80_rule', choices=['80_rule', 'jaro_v1'], help='Strategy to run (default: 80_rule)')
    args = parser.parse_args()
    
    # Determine VA Percent based on Strategy
    if args.strategy == 'jaro_v1':
        va_p = 0.45
    else:
        va_p = 0.70
        
    df = preprocess_data(DATA_PATH, va_percent=va_p)
    
    # Filter Date
    start_date = "2025-04-01"
    end_date = "2025-07-02"
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df.loc[mask]
    
    print(f"Test Rows: {len(subset)}")
    print(f"Strategy Selected: {args.strategy}")
    
    if args.strategy == 'jaro_v1':
        final_equity, trade_list, strat_name = run_strategy_jaro_v1(subset)
    else:
        final_equity, trade_list, strat_name = run_strategy_80_rule(subset)
    
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Total Trades: {len(trade_list)}")
    if len(trade_list) > 0:
        trades_df = pd.DataFrame(trade_list)
        print("First 5 Trades:")
        print(trades_df.head())
        print("\nPnL Stats:")
        print(trades_df.PnL.describe())
        
        plot_results(subset, trade_list, strat_name)
