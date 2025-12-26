import pandas as pd
import numpy as np
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# --- Configuration ---
DATA_PATH = r"C:\Users\geoff\.gemini\antigravity\scratch\project_repo\USDJPY 1M_CANDLESTICK DATA 2015-2025\USD_JPY_2015_07_2025_BID.csv"
VA_PERCENT = 0.70
INITIAL_CAPITAL = 10000
POSITION_SIZE = 1000  # Units
LEVERAGE = 50
SL_TICKS = 50
TP_TICKS = 100 
USE_OPPOSITE_VA_TP = True

def preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # Fast Parse
    # Use coerce to handle any errors, assuming mostly valid
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True, errors='coerce')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None)
    df.index.name = 'Date'
    df = df[~df.index.duplicated(keep='first')]
    
    # Cast
    cols = ['Open', 'High', 'Low', 'Close', 'volume']
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # Daily VA
    print("Calculating VA...")
    daily = df.resample('D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    daily['POC'] = (daily['High'] + daily['Low'] + daily['Close'])/3
    daily['Range'] = daily['High'] - daily['Low']
    daily['VAH'] = daily['POC'] + (daily['Range']*VA_PERCENT/2)
    daily['VAL'] = daily['POC'] - (daily['Range']*VA_PERCENT/2)
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

def run_backtest(df):
    print("Running Loop...")
    # Iterate
    # Iterate
    # Status: 'flat', 'long', 'short'
    status = 'flat'
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    
    trades = []
    equity = INITIAL_CAPITAL
    
    # Arrays for speed
    opens = df.Open.values
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
                exit_price = sl # Slippage? Assume limit
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
            # Criteria
            # Accepted
            if accepted[i]:
                # Open Above VA?
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
                    
    return equity, trades

def plot_results(df, trades):
    print("Generating Chart...")
    
    # --- Metrics Calculation ---
    df_trades = pd.DataFrame(trades)
    equity_curve = []
    
    if len(trades) > 0:
        df_trades.sort_values('ExitTime', inplace=True)
        df_trades['CumulativePnL'] = df_trades['PnL'].cumsum()
        df_trades['Equity'] = INITIAL_CAPITAL + df_trades['CumulativePnL']
        
        # Drawdown
        running_peak = df_trades['Equity'].cummax()
        drawdown = running_peak - df_trades['Equity']
        drawdown_pct = (drawdown / running_peak) * 100
        max_dd_pct = drawdown_pct.max()
        max_dd_val = drawdown.max()
        
        # Stats
        total_pnl = df_trades['PnL'].sum()
        wins = df_trades[df_trades['PnL'] > 0]
        losses = df_trades[df_trades['PnL'] <= 0]
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        win_rate = (len(wins) / len(df_trades)) * 100
        total_trades = len(df_trades)
        
        # Prepare Equity Curve Data (Start + Exits)
        # We add the start point
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
        subplot_titles=('Price Action', 'Equity Curve', 'Trade List')
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
        pnl_text = [f"PnL: {t['PnL']:.2f}" for t in trades]
        
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

    # 4. Equity Curve (Row 2)
    fig.add_trace(go.Scatter(
        x=eq_dates, y=eq_values,
        mode='lines', 
        name='Equity',
        line=dict(color='cyan', width=2),
        fill='tozeroy'
    ), row=2, col=1)
    
    # 5. Trade Table (Row 3)
    if len(trades) > 0:
        # Format for table
        t_entry_time = [t['EntryTime'].strftime('%Y-%m-%d %H:%M') for t in trades]
        t_type = [t['Type'] for t in trades]
        t_entry_p = [f"{t['Entry']:.3f}" for t in trades]
        t_exit_time = [t['ExitTime'].strftime('%Y-%m-%d %H:%M') for t in trades]
        t_exit_p = [f"{t['Exit']:.3f}" for t in trades]
        t_pnl = [f"{t['PnL']:.2f}" for t in trades]
        t_reason = [t['Reason'] for t in trades]
        
        # Color PnL
        fill_colors = ['palegreen' if t['PnL'] > 0 else 'lightpink' for t in trades]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['Entry Time', 'Type', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL', 'Reason'],
                fill_color='grey',
                align='left',
                font=dict(color='white')
            ),
            cells=dict(
                values=[t_entry_time, t_type, t_entry_p, t_exit_time, t_exit_p, t_pnl, t_reason],
                fill_color=[['black']*len(trades), ['black']*len(trades), ['black']*len(trades), ['black']*len(trades), ['black']*len(trades), fill_colors, ['black']*len(trades)],
                align='left',
                font=dict(color=['white', 'white', 'white', 'white', 'white', 'black', 'white'])
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
        title=dict(text=f'Backtest Results (80% Rule)<br><sup>{stats_text}</sup>', x=0.5),
        yaxis_title='USD/JPY',
        yaxis2_title='Equity ($)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=1000  # Increased height for table
    )
    
    # Save and Open
    filename = "backtest_chart.html"
    fig.write_html(filename)
    print(f"Chart saved to {filename}")
    webbrowser.open('file://' + os.path.realpath(filename))

if __name__ == "__main__":
    df = preprocess_data(DATA_PATH)
    
    # Filter Date
    start_date = "2025-04-01"
    end_date = "2025-07-02"
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df.loc[mask]
    
    print(f"Test Rows: {len(subset)}")
    final_equity, trade_list = run_backtest(subset)
    
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Total Trades: {len(trade_list)}")
    if len(trade_list) > 0:
        trades_df = pd.DataFrame(trade_list)
        print("First 5 Trades:")
        print(trades_df.head())
        print("\nPnL Stats:")
        print(trades_df.PnL.describe())
        
        # Plot
        plot_results(subset, trade_list)
