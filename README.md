# 80% Value Area Rule Strategy (USD/JPY)

This repository contains the Pine Script strategy and Python backtesting tools for the "80% Value Area Rule" on USD/JPY.

## Project Structure

- **`strategies/geoffs_strategy.pine`**: The core TradingView Strategy (v7).
- **`vector_backtest.py`**: Fast local backtester using Pandas and Plotly (Generates `backtest_chart.html`).
- **`fetch_data.py`**: Utility to download fresh USD/JPY data from AlphaVantage.
- **`analyze_october.py`**: Script used to analyze market conditions (Trend vs Range).

## 1. Pine Script Strategy (`geoffs_strategy.pine`)

A mean-reversion strategy based on the 80% Rule.

### Key Features (v7)
- **Trend Filter**: Uses 200 EMA to avoid fighting strong trends. (Default: **Counter Trend** mode).
- **Risk Management**:
    - **Circuit Breaker**: Stops trading if Max Daily Loss hits **1%**.
    - **Trailing Stop**: Locks in profits during volatility.
    - **Force Close**: Closes all positions at 16:55 (Session End).
- **Time Constraints**: Trading Window **09:30 - 15:00**. PM Profit Secure closes winners after 3 PM.

### Setup
1. Copy content of `strategies/geoffs_strategy.pine`.
2. Paste into TradingView Pine Editor.
3. Recommended Timeframe: **30 Minute**.

## 2. Python Backtester (`vector_backtest.py`)

A high-performance local backtester that validates the strategy logic.

### Usage
```bash
pip install -r requirements.txt
python vector_backtest.py
```

### Output
- Generates `backtest_chart.html`: An interactive dashboard with:
    -   **Price Chart**: Candlesticks + VAH/VAL logic.
    -   **Equity Curve**: Visualizing account growth.
    -   **Trade List**: Detailed table of every execution.

## 3. Data Augmentation (`fetch_data.py`)

Tools to fetch fresh data from AlphaVantage.

### Data Sources & Setup
To run the local backtest, you need the historical data.

1.  **Base Data (Kaggle)**:
    *   Download the **USD/JPY 1-Minute Candlestick Data (2015-2025)** from Kaggle:
    *   [Link to Dataset](https://www.kaggle.com/datasets/gauravox/usdjpy-1-minute-forex-candlestick-data-20152025)
    *   **Action**: Unzip and place `USD_JPY_2015_07_2025_BID.csv` into the folder: `USDJPY 1M_CANDLESTICK DATA 2015-2025/`.

2.  **Augmentation (AlphaVantage)**:
    *   Get a Free API Key: [AlphaVantage Support](https://www.alphavantage.co/support/#api-key)
    *   API Endpoint Used: `FX_DAILY` (Free) or `FX_INTRADAY` (Premium).

### Running the Augmentation
1.  Open `fetch_data.py`.
2.  Replace `API_KEY = "YOUR_ALPHAVANTAGE_API_KEY"` with your key.
3.  Run:
    ```bash
    python fetch_data.py
    ```
4.  This will generate `USD_JPY_DAILY_AUGMENTED.csv` (or merged data if configured).

---
**License**: Private
