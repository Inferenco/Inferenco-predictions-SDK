# Stohn Coin Test Commands

Run these commands from the `prediction_sdk` directory to verify forecasts for Stohn Coin (SOH) across different timeframes.

## Short-Term Forecasts
*   **15 Minutes:**
    ```bash
    cargo run --example stohn_15m
    ```
*   **1 Hour:**
    ```bash
    cargo run --example stohn_1h
    ```
*   **4 Hours:**
    ```bash
    cargo run --example stohn_4h
    ```

## Long-Term Forecasts
*   **1 Month:**
    ```bash
    cargo run --example stohn_1m
    ```
*   **3 Months:**
    ```bash
    cargo run --example stohn_3m
    ```
*   **1 Year:**
    ```bash
    cargo run --example stohn_1y
    ```

## Chart responses

Set `chart=true` on your payload to receive a `chart.history` array of OHLC
candles alongside the forecast. Each entry contains `timestamp`, `open`,
`high`, `low`, `close`, and an optional `volume` if the upstream endpoint
provides it. The SDK will call CoinGecko's `/ohlc` endpoint when available or
derive the candle values from the fetched market chart data.

## Backtesting Examples

Validate prediction accuracy and trading strategies using historical data.

*   **Simple Backtest (Walk-Forward Accuracy):**
    ```bash
    cargo run --example simple_backtest
    ```
    Measures MAE, RMSE, MAPE, and directional accuracy.

*   **Trading Strategy Backtest:**
    ```bash
    cargo run --example trading_strategy_backtest
    ```
    Simulates trading based on ML predictions, tracks P&L, win rate, and drawdown.

*   **ML Validation Backtest:**
    ```bash
    cargo run --example ml_validation_backtest
    ```
    Validates prediction interval coverage and reliability calibration.
