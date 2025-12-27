import argparse
import sys
import pandas as pd
from engine.backtester import VectorBacktester
from strategies.rule_80 import Rule80Strategy
from strategies.jaro_v1 import JaroV1Strategy
from engine import db_client

def main():
    parser = argparse.ArgumentParser(description='Run Trading Strategy Backtest')
    parser.add_argument('--strategy', type=str, default='80_rule', choices=['80_rule', 'jaro_v1'], help='Strategy to run (default: 80_rule)')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start Date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31', help='End Date (YYYY-MM-DD)')
    parser.add_argument('--init-db', action='store_true', help='Initialize DB from CSV (if needed)')
    parser.add_argument('--csv-path', type=str, help='Path to CSV for DB initialization')
    
    args = parser.parse_args()

    # 1. DB Initialization (Optional)
    if args.init_db:
        if not args.csv_path:
            print("Error: --csv-path required for initialization.")
            sys.exit(1)
        db_client.init_db()
        db_client.load_csv_to_db(args.csv_path)

    # 2. Load Data
    print(f"Loading Data ({args.start} to {args.end})...")
    try:
        df = db_client.get_market_data(args.start, args.end)
    except Exception as e:
        print(f"Error loading data: {e}. Ensure DB is populated (use --init-db) or path is correct.")
        sys.exit(1)

    if len(df) == 0:
        print("No data found for the specified range.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows.")

    # 3. Initialize Strategy
    if args.strategy == 'jaro_v1':
        strategy = JaroV1Strategy()
    else:
        strategy = Rule80Strategy()

    # 4. Run Backtest
    backtester = VectorBacktester()
    # Note: Strategy.run() returns (equity, trades, df_with_indicators)
    final_equity, trades, enriched_df = strategy.run(df)
    
    print(f"Final Equity: {final_equity:.2f}")
    if trades:
        print(f"Total Trades: {len(trades)}")
        backtester.plot_results(enriched_df, trades, strategy.name)
    else:
        print("No trades generated.")

if __name__ == "__main__":
    main()
