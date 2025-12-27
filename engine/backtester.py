import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import engine.db_client as db_client # Adapted import

class VectorBacktester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def plot_results(self, df, trades, strategy_name):
        print(f"Generating Chart for {strategy_name}...")
        
        # --- Optimization: Resampling for Large Datasets ---
        THRESHOLD_ROWS = 100000 
        
        if len(df) > THRESHOLD_ROWS:
            print(f"Dataset too large ({len(df)} rows). Resampling to 4H for visualization...")
            df_resampled = df.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'PrevVAH': 'first', 'PrevVAL': 'first' 
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
            df_trades['PnL'] = pd.to_numeric(df_trades['PnL'])
            df_trades['CumulativePnL'] = df_trades['PnL'].cumsum()
            df_trades['Equity'] = self.initial_capital + df_trades['CumulativePnL']
            
            # Stats
            total_pnl = df_trades['PnL'].sum()
            wins = df_trades[df_trades['PnL'] > 0]
            
            # Save to DB (using the new client location)
            # Ensure db_client is initialized or handled
            try:
                db_client.save_backtest_run(strategy_name, "Config", trades) 
            except Exception as e:
                print(f"Warning: Could not save to DB: {e}")
            
            eq_dates = [df.index[0]] + df_trades['ExitTime'].tolist()
            eq_values = [self.initial_capital] + df_trades['Equity'].tolist()
            
            win_rate = (len(wins) / len(df_trades)) * 100
        else:
            total_pnl = 0
            win_rate = 0
            eq_dates = [df.index[0]]
            eq_values = [self.initial_capital]

        # --- Plotting ---
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.5, 0.2, 0.3],
            specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
            subplot_titles=(f'Price Action ({strategy_name}) {CHART_TITLE_SUFFIX}', 'Equity Curve', 'Trade List')
        )
        
        # 1. Price Candle
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
            low=plot_df['Low'], close=plot_df['Close'], name='Price'
        ), row=1, col=1)
        
        # 2. Indicators
        if 'PrevVAH' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['PrevVAH'], line=dict(color='green', width=1), name='VAH'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['PrevVAL'], line=dict(color='red', width=1), name='VAL'), row=1, col=1)
        
        # 3. Trades
        if len(trades) > 0:
            long_entries_t = [t['EntryTime'] for t in trades if t['Type'] == 'Long']
            long_entries_p = [t['Entry'] for t in trades if t['Type'] == 'Long']
            short_entries_t = [t['EntryTime'] for t in trades if t['Type'] == 'Short']
            short_entries_p = [t['Entry'] for t in trades if t['Type'] == 'Short']
            exits_t = [t['ExitTime'] for t in trades]
            exits_p = [t['Exit'] for t in trades]
            
            if long_entries_t:
                fig.add_trace(go.Scatter(x=long_entries_t, y=long_entries_p, mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Long Entry'), row=1, col=1)
            if short_entries_t:
                fig.add_trace(go.Scatter(x=short_entries_t, y=short_entries_p, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Short Entry'), row=1, col=1)
            if exits_t:
                fig.add_trace(go.Scatter(x=exits_t, y=exits_p, mode='markers', marker=dict(symbol='x', color='blue', size=8), name='Exit'), row=1, col=1)

        # 4. Equity
        fig.add_trace(go.Scatter(x=eq_dates, y=eq_values, mode='lines', name='Equity', line=dict(color='cyan'), fill='tozeroy'), row=2, col=1)
        
        # 5. Table
        if len(trades) > 0:
            t_entry_time = [t['EntryTime'].strftime('%Y-%m-%d %H:%M') for t in trades]
            t_type = [t['Type'] for t in trades]
            t_pnl = [f"{t['PnL']:.2f}" for t in trades]
            
            fig.add_trace(go.Table(
                header=dict(values=['Entry Time', 'Type', 'PnL'], fill_color='grey', font=dict(color='white')),
                cells=dict(values=[t_entry_time, t_type, t_pnl], fill_color='black', font=dict(color='white'))
            ), row=3, col=1)

        filename = f"backtest_chart_{strategy_name.replace(' ', '_')}.html"
        fig.write_html(filename)
        print(f"Chart saved to {filename}")
        # webbrowser.open('file://' + os.path.realpath(filename)) # Disabled for non-interactive run
