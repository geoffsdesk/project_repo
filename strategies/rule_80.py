import pandas as pd
from strategies.base_strategy import BaseStrategy
from engine.indicators import calculate_value_areas, calculate_acceptance

class Rule80Strategy(BaseStrategy):
    def __init__(self, sl_ticks=50, tp_ticks=100, use_opposite_va=True):
        super().__init__("80% Rule")
        self.sl_ticks = sl_ticks
        self.tp_ticks = tp_ticks
        self.use_opposite_va = use_opposite_va

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Preprocess / Indicators
        df = calculate_value_areas(df, va_percent=0.70)
        df = calculate_acceptance(df, period='30min')
        
        # 2. Logic
        # We don't signal every bar here generally, we just identify conditions.
        # But 'engine' might expect a 'Signal' column?
        # The vector backtester in this project iterates row-by-row for trade mgmt.
        # So we just ensure the columns are there for the loop to read.
        
        # Required columns for this strategy: 
        # PrevVAH, PrevVAL, TodayOpen, Accepted
        
        return df

    def run(self, df: pd.DataFrame, initial_capital=10000, position_size=1000) -> tuple:
        print(f"Running Strategy: {self.name}...")
        df = self.generate_signals(df)
        
        status = 'flat'
        entry_price = 0.0
        sl = 0.0
        tp = 0.0
        trades = []
        equity = initial_capital
        
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
                    pnl = (exit_price - entry_price) * position_size
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'SL'})
                    status = 'flat'
                elif highs[i] >= tp:
                    exit_price = tp
                    pnl = (exit_price - entry_price) * position_size
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'TP'})
                    status = 'flat'
            elif status == 'short':
                 if highs[i] >= sl:
                    exit_price = sl
                    pnl = (entry_price - exit_price) * position_size
                    equity += pnl
                    trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': 'Short', 'Entry': entry_price, 'Exit': exit_price, 'PnL': pnl, 'Reason': 'SL'})
                    status = 'flat'
                 elif lows[i] <= tp:
                    exit_price = tp
                    pnl = (entry_price - exit_price) * position_size
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
                        sl = price + self.sl_ticks * 0.001
                        tp = prev_val[i] if self.use_opposite_va else price - self.tp_ticks * 0.001
                        status = 'short'
                    elif is_below:
                        entry_price = price
                        entry_time = times[i]
                        sl = price - self.sl_ticks * 0.001
                        tp = prev_vah[i] if self.use_opposite_va else price + self.tp_ticks * 0.001
                        status = 'long'
                        
        return equity, trades, df

    def get_pine_script(self) -> str:
        return "// Pine Script for 80% Rule (TODO)"
