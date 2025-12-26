
def plot_results(df, trades):
    print("Generating Chart...")
    
    # Create Figure
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # 1. Price Candle
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # 2. Indicators (VAH, VAL)
    # Filter to avoid too many points? Plotly handles 100k ok-ish, but let's be safe. 
    # Actually for 1m data over 6 months (~180k pts), it might be heavy.
    # But let's try full reso first.
    
    fig.add_trace(go.Scatter(x=df.index, y=df['PrevVAH'], mode='lines', line=dict(color='green', width=1), name='VAH'))
    fig.add_trace(go.Scatter(x=df.index, y=df['PrevVAL'], mode='lines', line=dict(color='red', width=1), name='VAL'))
    
    # 3. Trades
    # Separate Long/Short Entries and Exits
    long_entries = [t for t in trades if t['Type'] == 'Long']
    short_entries = [t for t in trades if t['Type'] == 'Short']
    
    # Long Entries
    if long_entries:
        le_times = [t['ExitTime'] for t in long_entries] # Wait, EntryTime is not stored in trade list properly?
        # Ah, 'ExitTime' is there. 'Entry' price is there. But 'EntryTime' is missing from the trade dict in run_backtest!
        # I need to fix run_backtest to store EntryTime.
        pass
        
    # Let's fix loop to store Entry Time first.
    pass

