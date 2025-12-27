import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from engine.indicators import calculate_value_areas, calculate_acceptance

class JaroV1Strategy(BaseStrategy):
    def __init__(self, sl_ticks=50):
        super().__init__("Jaro V1")
        self.sl_ticks = sl_ticks
        # Jaro V1 specific
        self.time_entry_limit = 900     # 15:00 NY? No, 900 mins from midnight = 15:00
        self.time_eod_flush = 960       # 16:00
        self.time_hard_close = 1005     # 16:45

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calculate_value_areas(df, va_percent=0.45) # Jaro uses 0.45
        df = calculate_acceptance(df, period='30min')
        
        # Add Jaro targets
        df['VASpread'] = df['PrevVAH'] - df['PrevVAL']
        df['LongTargetT1'] = df['PrevVAL'] + (df['VASpread'] * 0.75)
        df['ShortTargetT1'] = df['PrevVAH'] - (df['VASpread'] * 0.75)
        
        # Time calc
        # Assuming df has index localized or we handle in backtester.
        # Ideally we add 'MinutesFromMidnight' here.
        # But Timezone conversion is tricky without knowing source tz.
        # Assuming UTC source, converting to NY.
        
        if df.index.tz is None:
             ts_utc = df.index.tz_localize('UTC')
        else:
             ts_utc = df.index
             
        ts_ny = ts_utc.tz_convert('America/New_York')
        df['MinutesFromMidnight'] = ts_ny.hour * 60 + ts_ny.minute
        
        return df

    def run(self, df: pd.DataFrame, initial_capital=10000, position_size=1000) -> tuple:
        print(f"Running Strategy: {self.name}...")
        df = self.generate_signals(df) # Ensure indicators are present
        
        status = 'flat'
        trades_today = 0
        tp_candle_level = np.nan
        entry_price = 0.0
        sl = 0.0
        tp = 0.0
        trades = []
        equity = initial_capital
        
        dates = df.index.date
        day_indices = np.where(dates[1:] != dates[:-1])[0] + 1
        day_change_mask = np.zeros(len(df), dtype=bool)
        day_change_mask[day_indices] = True
        
        # Cache columns for speed
        highs = df.High.values
        lows = df.Low.values
        closes = df.Close.values
        times = df.index
        prev_vah = df.PrevVAH.values
        prev_val = df.PrevVAL.values
        today_open = df.TodayOpen.values
        accepted = df.Accepted.values
        long_target_t1 = df.LongTargetT1.values
        short_target_t1 = df.ShortTargetT1.values
        minutes_from_midnight = df.MinutesFromMidnight.values
        
        n = len(df)
        entry_time = None
        
        for i in range(n):
            if day_change_mask[i]:
                trades_today = 0
                tp_candle_level = np.nan
                    
            current_minutes = minutes_from_midnight[i]
            price = closes[i]
            
            if status != 'flat':
                if current_minutes >= self.time_hard_close:
                     pnl = (price - entry_price) * position_size if status == 'long' else (entry_price - price) * position_size
                     equity += pnl
                     trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': status.capitalize(), 'Entry': entry_price, 'Exit': price, 'PnL': pnl, 'Reason': 'EOD Risk Cut'})
                     status = 'flat'
                elif current_minutes >= self.time_eod_flush:
                    open_pnl = (price - entry_price) * position_size if status == 'long' else (entry_price - price) * position_size
                    if open_pnl > 0:
                        equity += open_pnl
                        trades.append({'EntryTime': entry_time, 'ExitTime': times[i], 'Type': status.capitalize(), 'Entry': entry_price, 'Exit': price, 'PnL': open_pnl, 'Reason': 'EOD Profit Flush'})
                        status = 'flat'
                
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
                        if trades_today == 1: tp_candle_level = highs[i]

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
                        if trades_today == 1: tp_candle_level = lows[i]
                            
            can_enter = current_minutes < self.time_entry_limit
            
            if status == 'flat' and can_enter:
                if trades_today == 0 and accepted[i]:
                    if today_open[i] > prev_vah[i]:
                        status = 'short'
                        entry_price = price
                        entry_time = times[i]
                        sl = prev_vah[i] + self.sl_ticks * 0.001
                        tp = short_target_t1[i]
                        trades_today = 1
                    elif today_open[i] < prev_val[i]:
                        status = 'long'
                        entry_price = price
                        entry_time = times[i]
                        sl = prev_val[i] - self.sl_ticks * 0.001
                        tp = long_target_t1[i]
                        trades_today = 1
                        
                elif trades_today == 1 and not np.isnan(tp_candle_level):
                    if today_open[i] < prev_val[i] and price > tp_candle_level:
                         status = 'long'
                         entry_price = price
                         entry_time = times[i]
                         sl = prev_val[i] - self.sl_ticks * 0.001
                         tp = prev_vah[i]
                         trades_today = 2
                    elif today_open[i] > prev_vah[i] and price < tp_candle_level:
                        status = 'short'
                        entry_price = price
                        entry_time = times[i]
                        sl = prev_vah[i] + self.sl_ticks * 0.001
                        tp = prev_val[i]
                        trades_today = 2

        return equity, trades, df

    def get_pine_script(self) -> str:
        return "// Pine Script for Jaro V1 (TODO)"
