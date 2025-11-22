# Changelog

All notable changes to the Prediction SDK will be documented here.

## [Unreleased]

### Added
- **Backtesting Examples**: Three comprehensive examples for strategy validation
  - `simple_backtest.rs`: Walk-forward accuracy testing with MAE, RMSE, MAPE, and directional accuracy metrics
  - `trading_strategy_backtest.rs`: Trading simulation with P&L tracking, win rate, max drawdown, and comparison to buy-and-hold
  - `ml_validation_backtest.rs`: Statistical validation of ML prediction intervals and reliability calibration

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-11-21

### Added
- **Chart-Aware Forecasts**: Added `chart` flag to `ForecastRequest` to optionally retrieve historical candles and projection bands.
- **Chart Data Structures**: Introduced `ForecastResponse`, `ForecastChart`, `ChartCandle`, and `ForecastBandPoint` DTOs.
- **Monte Carlo Projections**: Updated long-term forecasting to generate per-day projection bands when chart data is requested.
- **New Examples**: Added `chart_btc_15m.rs` and `chart_eth_1y.rs` demonstrating how to fetch and display chart data.
- **Documentation**: Updated `README.md` and `DOCUMENTATION.md` with "Fetching Charts" guides and API details.

## [0.1.1] - 2025-11-20

### Changed
- **Unified Forecast Output**: Standardized output format for short and long-term forecasts, including consistent "Expected Price" and "AI Prediction" fields.
- **Improved Confidence Scoring**: Tuned confidence scaling factor (reduced from 25.0 to 15.0) to provide more realistic scores for volatile assets.
- **Batched History Ingestion**: Implemented `fetch_price_history_batched` to stitch together multiple 90-day chunks, allowing for longer historical context without hitting API limits.
- **Optimized Lookback**: Adjusted `SHORT_FORECAST_LOOKBACK_DAYS` to 90 days to balance ML context with API usage.
- **CI/CD Fixes**: Updated integration tests to correctly mock the `/market_chart/range` endpoint.

### Added
- **Individual Test Examples**: Added specific test files (`stohn_15m.rs` through `stohn_1y.rs`) for granular verification of different forecast horizons.
- **AI Prediction for Long-Term**: Exposed the raw Monte Carlo mean as an "AI Prediction" for long-term forecasts.

## [0.1.0] - 2025-01-20

### Added
- **Hybrid Forecasting Engine**: Ensemble approach combining statistical models, machine learning, and technical analysis
- **Multi-Horizon Forecasting**: Support for 8 time horizons from 15 minutes to 4 years
  - Short-term: 15m, 1h, 4h
  - Long-term: 1d, 3d, 1w, 1m, 3m, 6m, 1y, 4y
- **Machine Learning**: On-the-fly Support Vector Regressor (SVR) training with linear kernel
  - Features: Lagged log returns + normalized RSI
  - Training: Automatic using last 30+ price points
- **Technical Analysis**: Integration with `ta` crate for indicators
  - RSI (14-period)
  - MACD (12/26 EMA) divergence
  - Bollinger Bands (20-period, 2σ)
  - Trend strength metric
- **Two-Tier Caching System**:
  - API response cache with dynamic TTL (5min to 1h based on data window)
  - Forecast result cache with horizon-based TTL (3min to 12h)
  - Implementation using `moka` with per-key expiry
- **Rate Limiting**:
  - Token bucket rate limiter (8 requests/second)
  - Exponential backoff retry logic for 429 errors
  - Retry-After header support
  - Maximum 5 retries with intelligent backoff
- **Confidence Scoring**: Dynamic confidence calculation based on volatility and forecast horizon
- **Sentiment Integration**: Optional sentiment snapshots (news + social scores) for forecast adjustment
- **CoinGecko API Integration**: Full support for market_chart endpoint with both days and range queries
- **MCP Handler**: Ready-to-use handler for Model Context Protocol integration
- **Comprehensive Documentation**: API reference, architecture overview, and integration examples
- **CI/CD Pipeline**: GitHub Actions workflow with formatting, linting, building, and testing
- **Example Scripts**: Real-world usage examples demonstrating all features

### Features
- Async-first design using `tokio`
- Type-safe error handling with `thiserror`
- Zero-cost abstractions with Rust's ownership model
- Thread-safe caching with `Arc` wrappers
- Deterministic forecasting for reproducible research
- No external model files required (on-the-fly ML training)

### Dependencies
- `reqwest` (0.12) - HTTP client with rustls-tls
- `tokio` (1.x) - Async runtime
- `serde` / `serde_json` (1.x) - Serialization
- `chrono` (0.4) - Timestamp handling
- `smartcore` (0.3) - Machine learning
- `ta` (0.5) - Technical indicators
- `governor` (0.6) - Rate limiting
- `moka` (0.12) - Caching
- `statrs` (0.16) - Statistics
- `thiserror` (1.x) - Error handling

[0.1.0]: https://github.com/your-org/Inferenco-predictions-SDK/releases/tag/v0.1.0
