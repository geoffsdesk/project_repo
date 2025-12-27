from pynecore.lib import script, strategy, ta, input, plot, close, high, low, open_ as open, time

# Define the Strategy with settings matching the Pine Script version
@script.strategy(
    title="Geoff's 80% Rule Strategy (PyneCore)",
    shorttitle="80% Rule",
    overlay=True,
    initial_capital=10000,
    default_qty_type=strategy.cash,
    default_qty_value=10000,
    currency=strategy.USD
)
def rule_80_strategy():
    # --- Inputs ---
    # In PyneCore, inputs are defined similarly to Pine
    va_percent = input.float(0.70, title="Value Area Percent", minval=0.1, maxval=1.0)
    sl_ticks = input.int(50, title="Stop Loss (Ticks)")
    tp_ticks = input.int(100, title="Takeprofit (Ticks)")

    # --- Indicators ---
    # Calculate Value Area High/Low (Simplified logic for daily VAH/VAL from previous day)
    # Note: Full Volume Profile is complex in pure Pine/Pyne, so we approximate or presume pre-calculated
    # For this demo, we'll use a simplified High/Low range logic as a proxy if data isn't enriched
    # BUT, since our data has 'PrevVAH' columns, we can try to access them if custom data columns are supported.
    # Standard PyneCore works on OHLCV. 
    # We will implement the standard logic:
    
    # Logic: 
    # If open is inside yesterday's VA, and we break out --> Trade? 
    # Or 80% rule: If we open OUTSIDE and close INSIDE, we traverse to the other side.
    
    # Let's replicate the logic from rule_80.py roughly using native calls
    
    # 1. Get Previous Day's High/Low (Naive VA approximation for demo if no Volume Profile)
    # In a real PyneCore setup with this repo, we'd ensure 'VAH'/'VAL' are passed as series.
    # Assuming we calculate them or import them. 
    # For this example, let's use a placeholder approximation to make it runnable:
    pd_high = high[1] # Previous candle high (assuming daily)
    pd_low = low[1]   # Previous candle low
    
    # Placeholder VAH/VAL (e.g., 70% of range)
    range_ = pd_high - pd_low
    vah = pd_high - (range_ * (1 - va_percent) / 2)
    val = pd_low + (range_ * (1 - va_percent) / 2)
    
    # Plotting
    plot(vah, color="green", title="VAH")
    plot(val, color="red", title="VAL")
    
    # --- Logic ---
    # 80% Rule: Open is outside VA, then price enters VA
    
    # Check if we are "in" the value area
    in_value_area = (close < vah) and (close > val)
    
    # Setup: 
    # Long: Open < VAL, then Close > VAL
    long_signal = (open < val) and (close > val)
    
    # Short: Open > VAH, then Close < VAH
    short_signal = (open > vah) and (close < vah)
    
    # --- Execution ---
    if long_signal:
        strategy.entry("Long", strategy.long)
        # Set Exit (TP/SL)
        # PyneCore supports strategy.exit for TP/SL
        strategy.exit("Exit Long", "Long", profit=tp_ticks, loss=sl_ticks)
        
    if short_signal:
        strategy.entry("Short", strategy.short)
        strategy.exit("Exit Short", "Short", profit=tp_ticks, loss=sl_ticks)

# To run this:
# pynecore run strategies/pyne_strategy_80.py --symbol USDJPY --timeframe 1D
