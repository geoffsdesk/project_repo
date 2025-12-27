# 80% Value Area Rule Strategy (USD/JPY)

This repository contains Value Area trading strategies for USD/JPY, including the classic '80% Rule' and the new momentum-based 'Jaro V1'.

## Project Structure

- **`main.py`**: The primary entry point for running backtests and initializing the database.
- **`strategies/`**: Contains Python implementations and Pine Script versions of the strategies.
    - `rule_80.py` / `geoffs_strategy.pine`: The 80% Rule Strategy.
    - `jaro_v1.py` / `Jaro_v1.pine`: The Jaro V1 Momentum Strategy.
    - `pyne_strategy_80.py`: **[NEW]** PyneCore implementation of the 80% Rule.
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

## 3. Running Backtests (Python Engine)

Use `main.py` to run backtests with our custom vector engine.

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

## 4. Running With PyneCore (Native Pine Support)

This project now supports **PyneCore** (`pynesys-pynecore`), allowing you to run strategies written in Python syntax that mimics Pine Script 1:1.

**Usage:**
```bash
# Run the 80% Rule using the PyneCore engine
# Ensure you have data or use PyneCore's data fetcher
pynecore run strategies/pyne_strategy_80.py --symbol USDJPY --timeframe 1D
```

This ensures your logic matches TradingView's execution model exactly.

## 5. Data Analysis

You can write custom scripts to analyze the data effortlessly using the `engine.db_client`.

**Example: `analyze_october.py`**
This script fetches data for October 2025 from the DB and calculates volatility metrics.
```bash
python analyze_october.py
```

---
**License**: Private
