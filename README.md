# 80% Value Area Rule Strategy (USD/JPY)

This repository contains Value Area trading strategies for USD/JPY, including the classic '80% Rule' and the new momentum-based 'Jaro V1'.

## Project Structure

- **`strategies/geoffs_strategy.pine`**: The 80% Rule Strategy (v7).
- **`strategies/Jaro_v1.pine`**: The Jaro V1 Momentum Strategy.
- **`vector_backtest.py`**: Fast local backtester supporting multiple strategies.
- **`fetch_data.py`**: Utility to download fresh USD/JPY data from AlphaVantage.

## 1. Strategies

### A. 80% Rule (`geoffs_strategy.pine`)
A mean-reversion strategy based on the 80% Rule logic with 70% Value Area.
- **Key Features**: Trend filtering (200 EMA), 1% Max Daily Loss, Trailing Stop.
- **Default Parameter**: VA Percent = **70%**.

### B. Jaro V1 (`Jaro_v1.pine`)
A momentum and re-entry focused strategy.
- **Key Features**: 
    - **VA Percent**: **45%**.
    - **Re-Entry Logic**: Enters on momentum breaks of the candle that triggered the first Take Profit.
    - **Time Windows**: Entry before 3:00 PM EST, Hard Close at 4:45 PM EST.

## 2. Python Backtester (`vector_backtest.py`)

A high-performance local backtester that validates the strategy logic.

### Usage
```bash
pip install -r requirements.txt
```

**Run the 80% Rule Strategy (Default):**
```bash
python vector_backtest.py
# OR explicitly:
python vector_backtest.py --strategy 80_rule
```

**Run the Jaro V1 Strategy:**
```bash
python vector_backtest.py --strategy jaro_v1
```

### Output
- Generates `backtest_chart_[strategy_name].html`: An interactive dashboard with:
    -   **Price Chart**: Candlesticks + VAH/VAL logic.
    -   **Equity Curve**: Visualizing account growth.
    -   **Trade List**: Detailed table of every execution.

## 3. Data & Backtesting Notes
> [!IMPORTANT]
> **Mixed Data Resolution**
> The database contains a mix of resolutions:
> *   **Jan 2015 – July 2025**: High-fidelity **1-Minute** Data (Source: Kaggle).
> *   **Aug 2025 – Dec 2025**: **Daily** Data (Source: AlphaVantage Free Tier).
> 
> *Note: Backtesting over the late 2025 period uses Daily bars, so intraday stops/targets are approximated.*

### Database
The system now uses a SQLite database (`trading_data.db`) instead of raw CSVs for faster loading.
*   The database is pre-populated with the 10-year dataset.
*   **Performance**: For long backtests (e.g., 10 years), the charting engine automatically resamples price action to **4-Hour candles** to prevent browser crashes, while retaining precise trade markers.

### Advanced Usage
Run backtests on specific date ranges:
```bash
# Run Jaro V1 for the year 2024
python vector_backtest.py --strategy jaro_v1 --start 2024-01-01 --end 2024-12-31

# Run 80% Rule for Jan 2025
python vector_backtest.py --strategy 80_rule --start 2025-01-01 --end 2025-01-31
```

---
**License**: Private
