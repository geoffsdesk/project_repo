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

## 3. Data Setup

To run the local backtest, you need the historical data.

1.  **Base Data (Kaggle)**:
    *   Download the **USD/JPY 1-Minute Candlestick Data (2015-2025)** from Kaggle:
    *   [Link to Dataset](https://www.kaggle.com/datasets/gauravox/usdjpy-1-minute-forex-candlestick-data-20152025)
    *   **Action**: Unzip and place `USD_JPY_2015_07_2025_BID.csv` into the folder: `USDJPY 1M_CANDLESTICK DATA 2015-2025/`.

2.  **Augmentation (AlphaVantage) (Optional)**:
    *   Use `fetch_data.py` to get fresh daily data.

---
**License**: Private
