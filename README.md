# 80% Value Area Rule Strategy (USD/JPY)

This repository contains Value Area trading strategies for USD/JPY, including the classic '80% Rule' and the new momentum-based 'Jaro V1'.

## Project Structure

- **`main.py`**: The primary entry point for running backtests and initializing the database.
- **`strategies/`**: Contains Python implementations and Pine Script versions of the strategies.
    - `rule_80.py` / `geoffs_strategy.pine`: The 80% Rule Strategy.
    - `jaro_v1.py` / `Jaro_v1.pine`: The Jaro V1 Momentum Strategy.
- **`engine/`**: Core backtesting engine and database client.
- **`data/`**: Stores the SQLite database (`trading_data.db`) and raw CSV data.
- **`analyze_october.py`**: Example script for specific period analysis using the database.

## 1. Strategies

### A. 80% Rule
A mean-reversion strategy based on the 80% Rule logic with 70% Value Area.
- **Key Features**: Trend filtering, 1% Max Daily Loss, Trailing Stop.
- **Default Parameter**: VA Percent = **70%**.

### B. Jaro V1
A momentum and re-entry focused strategy.
- **Key Features**: 
    - **VA Percent**: **45%**.
    - **Re-Entry Logic**: Enters on momentum breaks.
    - **Time Windows**: Specific entry/exit windows (NY Session).

## 2. Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Initialize Database** (First time only):
Before running backtests, you need to populate the local SQLite database from your CSV data source.
```bash
python main.py --init-db --csv-path "data/USD_JPY_DAILY_AUGMENTED.csv"
```
*Note: Ensure your CSV file follows standard OHLCV format.*

## 3. Running Backtests

Use `main.py` to run backtests. The system defaults to the 80% Rule strategy.

**Run the 80% Rule Strategy:**
```bash
python main.py
# OR
python main.py --strategy 80_rule
```

**Run the Jaro V1 Strategy:**
```bash
python main.py --strategy jaro_v1
```

### Custom Date Ranges
You can specify custom start and end dates:
```bash
python main.py --strategy jaro_v1 --start 2024-01-01 --end 2024-12-31
```

### Output
- **Console**: Summary stats (Final Equity, Total Trades).
- **Charts**: Generates `backtest_chart_[strategy_name].html`. Open this file in your browser to view:
    - Interactive Price Chart with VAH/VAL.
    - Equity Curve.
    - Detailed Trade List table.
- **Database**: Results are saved to the `backtest_runs` and `trades` tables in `trading_data.db`.

## 4. Data Analysis

You can write custom scripts to analyze the data effortlessly using the `engine.db_client`.

**Example: `analyze_october.py`**
This script fetches data for October 2025 from the DB and calculates volatility metrics.
```bash
python analyze_october.py
```

## 5. Notes
> [!IMPORTANT]
> **Data Resolution**
> The system supports both minute and daily data. For long backtests (e.g., 10+ years), the visualization automatically resamples to **4-Hour candles** to improve performance, while the backtest logic runs on the highest resolution available.

---
**License**: Private
