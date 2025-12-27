import sqlite3
import pandas as pd
import os

DB_NAME = "trading_data.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Market Data Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp DATETIME PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    
    # Backtest Runs Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            config TEXT
        )
    ''')
    
    # Trades Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            entry_time DATETIME,
            exit_time DATETIME,
            type TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            reason TEXT,
            FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_csv_to_db(filepath):
    """
    Loads CSV data into the market_data table.
    Skips if data already exists to avoid duplication for this simple implementation.
    """
    print("Checking Database for existing data...")
    conn = get_db_connection()
    
    # Check count
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM market_data")
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"Database already has {count} rows. Skipping CSV load.")
        conn.close()
        return

    print(f"Loading CSV {filepath} to Database (this may take a while for large files)...")
    # Read CSV using Logic from vector_backtest's original preprocess but raw
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # Standardize Columns
    # Assuming standard Kaggle format map
    cols_map = {}
    for c in df.columns:
        lower_c = c.lower()
        if 'time' in lower_c: cols_map[c] = 'timestamp'
        elif 'open' in lower_c: cols_map[c] = 'open'
        elif 'high' in lower_c: cols_map[c] = 'high'
        elif 'low' in lower_c: cols_map[c] = 'low'
        elif 'close' in lower_c: cols_map[c] = 'close'
        elif 'vol' in lower_c: cols_map[c] = 'volume'
        
    df.rename(columns=cols_map, inplace=True)
    
    # Parse Dates
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True, errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    
    # Ensure raw types
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df.dropna(inplace=True)
    
    # Insert
    # SQLite is faster with 'to_sql' but chunksize matters
    df.to_sql('market_data', conn, if_exists='append', index=False, chunksize=10000)
    print("Data loaded to Database successfully.")
    conn.close()

def get_market_data(start_date=None, end_date=None):
    """
    Reads data from DB into DataFrame
    """
    conn = get_db_connection()
    query = "SELECT * FROM market_data"
    params = []
    
    if start_date and end_date:
        query += " WHERE timestamp >= ? AND timestamp <= ?"
        params = [start_date, end_date]
    
    query += " ORDER BY timestamp ASC"
    
    print("Fetching data from DB...")
    df = pd.read_sql(query, conn, params=params, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None) # Make naive for backtester compatibility if needed, but keeping consistent with prev code
    
    # Rename simple lower case back to Capitalized for backtester compatibility
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    conn.close()
    return df

def save_backtest_run(strategy_name, config_str, trades_list):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO backtest_runs (strategy_name, config) VALUES (?, ?)", (strategy_name, config_str))
    run_id = cursor.lastrowid
    
    if trades_list:
        # Prepare list of tuples
        # trade dict: entry_time, exit_time, type, entry, exit, pnl, reason
        data_to_insert = []
        for t in trades_list:
            data_to_insert.append((
                run_id,
                t['EntryTime'].isoformat() if hasattr(t['EntryTime'], 'isoformat') else t['EntryTime'],
                t['ExitTime'].isoformat() if hasattr(t['ExitTime'], 'isoformat') else t['ExitTime'],
                t['Type'],
                t['Entry'],
                t['Exit'],
                t['PnL'],
                t['Reason']
            ))
        
        cursor.executemany('''
            INSERT INTO trades (run_id, entry_time, exit_time, type, entry_price, exit_price, pnl, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
    
    conn.commit()
    conn.close()
    print(f"Backtest results saved with Run ID: {run_id}")
