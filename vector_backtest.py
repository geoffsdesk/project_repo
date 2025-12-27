import pandas as pd
import numpy as np
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import argparse
import db_utils

# --- Configuration ---
INITIAL_CAPITAL = 10000
POSITION_SIZE = 1000  # Units
LEVERAGE = 50
SL_TICKS = 50
TP_TICKS = 100 
USE_OPPOSITE_VA_TP = True

def preprocess_data(va_percent=0.70, start_date=None, end_date=None):
    # Initialize DB (Just in case)
    db_utils.init_db()
    
    # Fetch from DB
    print(f"Loading data from DB (Start: {start_date}, End: {end_date})...")
    df = db_utils.get_market_data(start_date, end_date)
    
    if len(df) == 0:
        print("No data found in DB for range. Please ensure database is populated.")
        sys.exit(1)

    # Note: DB Utils returns Capitalized Columns: Open, High, Low, Close, Volume
    # Index is Timestamp (Naive)
    
    # Sort index just in case
    df.sort_index(inplace=True)

    # Daily VA Calculation
    print("Calculating VA...")
    daily = df.resample('D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    daily['POC'] = (daily['High'] + daily['Low'] + daily['Close'])/3
    daily['Range'] = daily['High'] - daily['Low']
    daily['VAH'] = daily['POC'] + (daily['Range']*va_percent/2)
    daily['VAL'] = daily['POC'] - (daily['Range']*va_percent/2)
    daily['PrevVAH'] = daily['VAH'].shift(1)
    daily['PrevVAL'] = daily['VAL'].shift(1)
    daily['DateOnly'] = daily.index.date
    
    # Map Daily Levels
    print("Mapping Daily Levels...")
    df['DateOnly'] = df.index.date
    daily_indexed_by_date = daily.set_index('DateOnly')
    
    df['PrevVAH'] = df['DateOnly'].map(daily_indexed_by_date['PrevVAH'])
    df['PrevVAL'] = df['DateOnly'].map(daily_indexed_by_date['PrevVAL'])
    
    # Today Open
    day_open_map = daily.set_index('DateOnly')['Open']
    df['TodayOpen'] = df['DateOnly'].map(day_open_map)
    
    # Acceptance (30m)
    print("Calculating Acceptance...")
    res30 = df.resample('30min').agg({'Close':'last'})
    res30['DateOnly'] = res30.index.date
    res30['PrevVAH'] = res30['DateOnly'].map(daily_indexed_by_date['PrevVAH'])
    res30['PrevVAL'] = res30['DateOnly'].map(daily_indexed_by_date['PrevVAL'])
    
    res30['IsInside'] = (res30['Close'] < res30['PrevVAH']) & (res30['Close'] > res30['PrevVAL'])
    res30['Accepted'] = res30['IsInside'].shift(1) & res30['IsInside'].shift(2)
    
    df['Time30m'] = df.index.floor('30min')
    df['Accepted'] = df['Time30m'].map(res30['Accepted']).fillna(False).astype(bool)
    
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
        
        if status == 'flat':
            if accepted[i]:
                is_above = today_open[i] > prev_vah[i]
                is_below = today_open[i] < prev_val[i]
                if is_above:
                    entry_price = price
                    entry_time = times[i]
                    sl = price + SL_TICKS * 0.001
                    tp = prev_val[i] if USE_OPPOSITE_VA_TP else price - TP_TICKS * 0.001
                    status = 'short'
                elif is_below:
                    entry_price = price
                    entry_time = times[i]
                    sl = price - SL_TICKS * 0.001
                    tp = prev_vah[i] if USE_OPPOSITE_VA_TP else price + TP_TICKS * 0.001
                    status = 'long'
                    
    return equity, trades, "80% Rule"

def run_strategy_jaro_v1(df):
    print("Running Strategy: Jaro V1...")
    
    # UTC to NY
    ts_utc = df.index.tz_localize('UTC')
    ts_ny = ts_utc.tz_convert('America/New_York')
    minutes_from_midnight = (ts_ny.hour * 60 + ts_ny.minute).values
    
    TIME_ENTRY_Limit = 900 
    TIME_EOD_FLUSH = 960 
    TIME_HARD_CLOSE = 1005
    
    status = 'flat'
    trades_today = 0
    tp_candle_level = np.nan
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    trades = []
    equity = INITIAL_CAPITAL
    
    dates = df.index.date
    # Day change detection
    day_indices = np.where(dates[1:] != dates[:-1])[0] + 1
    day_change_mask = np.zeros(len(df), dtype=bool)
    day_change_mask[day_indices] = True
    
    # Values
    highs = df.High.values
    lows = df.Low.values
    closes = df.Close.values
    times = df.index
    prev_vah = df.PrevVAH.values
    prev_val = df.PrevVAL.values
    today_open = df.TodayOpen.values
    accepted = df.Accepted.values
    
    va_spread = prev_vah - prev_val
    long_target_t1 = prev_val + (va_spread * 0.75)
    short_target_t1 = prev_vah - (va_spread * 0.75)
    
    n = len(df)
    entry_time = None
    
    for i in range(n):
        if day_change_mask[i]:
            trades_today = 0
            tp_candle_level = np.nan
                
        current_minutes = minutes_from_midnight[i]
        price = closes[i]
        
        if status != 'flat':
            if current_minutes >= TIME_HARD_CLOSE:
                 pnl = (price - entry_price) * POSITION_SIZE if status == 'long' else (entry_price - price) * POSITION_SIZE
                 equity += pnl
                 trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': status.capitalize(), 'Entry': entry_price, 'Exit': price, 'PnL': pnl, 'Reason': 'EOD Risk Cut'})
                 status = 'flat'
            elif current_minutes >= TIME_EOD_FLUSH:
                open_pnl = (price - entry_price) * POSITION_SIZE if status == 'long' else (entry_price - price) * POSITION_SIZE
                if open_pnl > 0:
                    equity += open_pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': status.capitalize(), 'Entry': entry_price, 'Exit': price, 'PnL': open_pnl, 'Reason': 'EOD Profit Flush'})
                    status = 'flat'
            
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
                    if trades_today == 1: tp_candle_level = highs[i]

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
                    if trades_today == 1: tp_candle_level = lows[i]
                        
        can_enter = current_minutes < TIME_ENTRY_Limit
        
        if status == 'flat' and can_enter:
            if trades_today == 0 and accepted[i]:
                if today_open[i] > prev_vah[i]:
                    status = 'short'
                    entry_price = price
                    entry_time = times[i]
                    sl = prev_vah[i] + SL_TICKS * 0.001
                    tp = short_target_t1[i]
                    trades_today = 1
                elif today_open[i] < prev_val[i]:
                    status = 'long'
                    entry_price = price
                    entry_time = times[i]
                    sl = prev_val[i] - SL_TICKS * 0.001
                    tp = long_target_t1[i]
                    trades_today = 1
                    
            elif trades_today == 1 and not np.isnan(tp_candle_level):
                if today_open[i] < prev_val[i] and price > tp_candle_level:
                     status = 'long'
                     entry_price = price
                     entry_time = times[i]
                     sl = prev_val[i] - SL_TICKS * 0.001
                     tp = prev_vah[i]
                     trades_today = 2
                elif today_open[i] > prev_vah[i] and price < tp_candle_level:
                    status = 'short'
                    entry_price = price
                    entry_time = times[i]
                    sl = prev_vah[i] + SL_TICKS * 0.001
                    tp = prev_val[i]
                    trades_today = 2

    return equity, trades, "Jaro V1"

def plot_results(df, trades, strategy_name):
    print(f"Generating Chart for {strategy_name}...")
    
    # --- Optimization: Resampling for Large Datasets ---
    THRESHOLD_ROWS = 100000 
    
    if len(df) > THRESHOLD_ROWS:
        # Switch to 4H resampling
        print(f"Dataset too large ({len(df)} rows). Resampling to 4H for visualization...")
        
        # Resample OHLC
        df_resampled = df.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'PrevVAH': 'first', 'PrevVAL': 'first' # VAH/VAL shouldn't change much intraday, taking first is safe approximation
        }).dropna()
        
        CHART_TITLE_SUFFIX = f"(4H Resampled - {len(df_resampled)} bars)"
        plot_df = df_resampled
    else:
        plot_df = df
        CHART_TITLE_SUFFIX = "(1M Precision)"

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
        
        # Save to DB
        db_utils.save_backtest_run(strategy_name, "Config", trades)
        
        eq_dates = [df.index[0]] + df_trades['ExitTime'].tolist()
        eq_values = [INITIAL_CAPITAL] + df_trades['Equity'].tolist()
    else:
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
        subplot_titles=(f'Price Action ({strategy_name}) {CHART_TITLE_SUFFIX}', 'Equity Curve', 'Trade List')
    )
    
    # 1. Price Candle (Resampled if needed)
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # 2. Indicators (VAH, VAL)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['PrevVAH'], mode='lines', line=dict(color='green', width=1), name='VAH'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['PrevVAL'], mode='lines', line=dict(color='red', width=1), name='VAL'), row=1, col=1)
    
    # 3. Trades Markers (PRECISE, Original Timestamps)
    # We plot these on the same x-axis. Plotly handles mixed granularities fine usually.
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
        f"<b>Drawdown:</b> {max_dd_pct:.2f}% | "
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
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start Date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31', help='End Date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    if args.strategy == 'jaro_v1':
        va_p = 0.45
    else:
        va_p = 0.70
    
    # Range
    start_date = args.start
    end_date = args.end
        
    df = preprocess_data(va_percent=va_p, start_date=start_date, end_date=end_date)
    
    print(f"Test Rows: {len(df)}")
    print(f"Strategy Selected: {args.strategy}")
    
    if args.strategy == 'jaro_v1':
        final_equity, trade_list, strat_name = run_strategy_jaro_v1(df)
    else:
        final_equity, trade_list, strat_name = run_strategy_80_rule(df)
    
    print(f"Final Equity: {final_equity:.2f}")
    if len(trade_list) > 0:
        plot_results(df, trade_list, strat_name)
