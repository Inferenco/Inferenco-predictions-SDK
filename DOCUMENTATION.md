# Inferenco Predictions SDK - Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Forecasting Engine](#forecasting-engine)
3. [API Reference](#api-reference)
4. [Caching Strategy](#caching-strategy)
5. [Rate Limiting](#rate-limiting)
6. [Configuration](#configuration)
7. [MCP Integration](#mcp-integration)

## Architecture Overview

```
prediction_sdk/
├── src/
│   ├── dto.rs            # Data structs (price points, forecast DTOs, error enums)
│   ├── helpers.rs        # Statistical helpers (moving averages, Monte Carlo, sentiment)
│   ├── analysis.rs       # Technical indicators (RSI, MACD, BB) + ML model (SVR)
│   ├── cache.rs          # Two-tier caching (API responses + forecasts)
│   ├── handler.rs        # MCP-ready handler
│   ├── impl.rs           # PredictionSdk implementation
│   ├── mod.rs            # Module exports
│   └── lib.rs            # Crate entry point
```

## Forecasting Engine

The SDK uses a **hybrid ensemble approach**:

### 1. Statistical Baseline
- **Moving Average**: Rolling window based on horizon
- **Volatility**: Log-return standard deviation
- **Monte Carlo**: Geometric Brownian motion paths for long horizons

### 2. Technical Analysis
- **RSI (14-period)**: Relative Strength Index for overbought/oversold signals
- **MACD (12/26 EMA)**: Momentum divergence
- **Bollinger Bands (20-period, 2σ)**: Volatility bands
- **Trend Strength**: Proxy based on MACD normalized by price

### 3. Machine Learning
- **Model**: Support Vector Regressor (SVR) with linear kernel
- **Features**: Lagged log returns (t-1, t-2) + normalized RSI
- **Training**: On-the-fly using last 30+ price points
- **Prediction**: Next-step log return, converted back to price

### 4. Ensemble
For short-term forecasts:
```
final_price = (statistical_forecast + ml_forecast) / 2
```

### 5. Confidence Scoring
```
confidence = 1.0 / (1.0 + 100 * volatility * sqrt(horizon_hours))
```

Confidence decreases with:
- Higher volatility
- Longer forecast horizon

## API Reference

### Core Types

#### `PredictionSdk`
```rust
// Default constructor (uses public CoinGecko API)
let sdk = PredictionSdk::new()?;

// Custom client + base URL
let client = reqwest::Client::new();
let sdk = PredictionSdk::with_client(client, Some("https://custom-api.example.com".to_string()));
```

#### `ForecastHorizon`
```rust
// Short-term (15m, 1h, 4h)
ForecastHorizon::Short(ShortForecastHorizon::OneHour)

// Long-term (1d, 3d, 1w, 1m, 3m, 6m, 1y, 4y)
ForecastHorizon::Long(LongForecastHorizon::OneMonth)
```

#### `SentimentSnapshot`
```rust
SentimentSnapshot {
    news_score: 0.65,    // -1.0 to 1.0
    social_score: 0.80,  // -1.0 to 1.0
}
```

### Methods

#### `forecast_with_fetch`
Fetches data + computes forecast in one call (uses cache).

```rust
let result = sdk.forecast_with_fetch(
    "bitcoin",
    "usd",
    ForecastHorizon::Short(ShortForecastHorizon::OneHour),
    Some(sentiment),
).await?;
```

#### `fetch_price_history`
Fetch raw price data (days lookback).

```rust
let history = sdk.fetch_price_history("bitcoin", "usd", 30).await?;
```

#### `fetch_price_history_range`
Fetch raw price data (date range).

```rust
let from = Utc::now() - Duration::days(90);
let to = Utc::now();
let history = sdk.fetch_price_history_range("bitcoin", "usd", from, to).await?;
```

#### `forecast`
Compute forecast from pre-fetched history (bypasses API cache).

```rust
let result = sdk.forecast(&history, horizon, sentiment).await?;
```

### Result Types

#### `ShortForecastResult`
```rust
{
    horizon: ShortForecastHorizon,
    expected_price: f64,              // Ensemble prediction
    confidence: f32,                  // 0.0 to 1.0
    decomposition: ForecastDecomposition,
    technical_signals: Option<TechnicalSignals>,
    ml_prediction: Option<f64>,       // Raw ML output (before ensemble)
}
```

#### `LongForecastResult`
```rust
{
    horizon: LongForecastHorizon,
    mean_price: f64,                  // Monte Carlo mean
    percentile_10: f64,               // Bearish scenario
    percentile_90: f64,               // Bullish scenario
    confidence: f32,
    technical_signals: Option<TechnicalSignals>,
    ml_prediction: Option<f64>,
}
```

#### `TechnicalSignals`
```rust
{
    rsi: f64,                         // 0-100
    macd_divergence: f64,             // EMA(12) - EMA(26)
    bollinger_width: f64,             // Normalized band width
    trend_strength: f64,              // MACD / price * 100
}
```

## Caching Strategy

### API Cache (CoinGecko responses)
TTL based on data window:
- **≤7 days**: 5 minutes
- **8-30 days**: 15 minutes
- **>30 days**: 1 hour

Cache key: `(asset_id, vs_currency, days)` or `(asset_id, vs_currency, from_ts, to_ts)`

### Forecast Cache (computed results)
TTL based on forecast horizon:

**Short-term:**
- **15 minutes**: 3 min TTL
- **1 hour**: 10 min TTL
- **4 hours**: 30 min TTL

**Long-term:**
- **1 day**: 2 hours
- **3 days**: 4 hours
- **1 week**: 6 hours
- **1+ month**: 12 hours

Cache key: `(asset_id, vs_currency, horizon, sentiment_hash)`

### Implementation
- **Backend**: `moka` with `Expiry` trait for per-key TTL
- **Capacity**: 1000 API entries, 500 forecast entries
- **Thread-safe**: `Arc<Cache>` for concurrent access

## Rate Limiting

### Token Bucket
- **Rate**: 8 requests/second (burst 8)
- **Library**: `governor` crate

### Retry Logic
On `429 Too Many Requests`:
1. Check `Retry-After` header
2. If present: wait specified duration
3. If missing: exponential backoff (500ms, 1s, 2s, 4s, 8s)
4. Max retries: 5
5. After max retries: return error

### Usage
Automatic - all API calls go through rate limiter before execution.

## Configuration

### CoinGecko API
```rust
// Default (public API)
let sdk = PredictionSdk::new()?;

// Custom base URL
let sdk = PredictionSdk::with_client(client, Some("https://pro-api.coingecko.com/api/v3".to_string()));

// With API key (add to client headers)
use reqwest::header::{HeaderMap, HeaderValue};
let mut headers = HeaderMap::new();
headers.insert("X-CG-PRO-API-KEY", HeaderValue::from_static("your-key-here"));

let client = reqwest::Client::builder()
    .default_headers(headers)
    .build()?;
let sdk = PredictionSdk::with_client(client, Some("https://pro-api.coingecko.com/api/v3".to_string()));
```

### Environment Variables
```bash
export COINGECKO_BASE_URL="https://custom-api.example.com"
export COINGECKO_API_KEY="your-api-key"
```

Then in code:
```rust
let base_url = std::env::var("COINGECKO_BASE_URL").ok();
let api_key = std::env::var("COINGECKO_API_KEY").ok();

let mut headers = HeaderMap::new();
if let Some(key) = api_key {
    headers.insert("X-CG-PRO-API-KEY", HeaderValue::from_str(&key)?);
}

let client = reqwest::Client::builder()
    .default_headers(headers)
    .build()?;

let sdk = match base_url {
    Some(url) => PredictionSdk::with_client(client, Some(url)),
    None => PredictionSdk::with_client(client, None),
};
```

## MCP Integration

### Handler Function
```rust
use prediction_sdk::run_prediction_handler;

let request_json = r#"{
  "asset_id": "bitcoin",
  "vs_currency": "usd",
  "horizon": { "short": "one_hour" },
  "sentiment": { "news_score": 0.1, "social_score": -0.05 }
}"#;

let result = run_prediction_handler(request_json).await?;
println!("{}", result); // JSON string
```

### Request Schema
```json
{
  "asset_id": "bitcoin",
  "vs_currency": "usd",
  "horizon": {
    // Either:
    "short": "fifteen_minutes" | "one_hour" | "four_hours"
    // Or:
    "long": "one_day" | "three_days" | "one_week" | "one_month" | 
            "three_months" | "six_months" | "one_year" | "four_years"
  },
  "sentiment": {  // Optional
    "news_score": 0.1,    // -1.0 to 1.0
    "social_score": -0.05 // -1.0 to 1.0
  }
}
```

### Error Handling
```rust
match run_prediction_handler(request_json).await {
    Ok(json) => {
        // Success: parse ForecastResult from json
    }
    Err(e) => {
        // PredictionError variants:
        // - Network(String)
        // - Serialization(String)
        // - InsufficientData
        // - TimeConversion
    }
}
```

## Dependencies

```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
governor = "0.6"
moka = { version = "0.12", features = ["future"] }
nonzero_ext = "0.3"
reqwest = { version = "0.12", features = ["json", "rustls-tls"], default-features = false }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
smartcore = { version = "0.3", default-features = true }
statrs = "^0.16"
ta = "0.5"
thiserror = "^1.0"
tokio = { version = "1", features = ["full"] }
ndarray = "0.15"
```

## Testing

```bash
# All tests
cargo test

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'

# Doc tests only
cargo test --doc

# With output
cargo test -- --nocapture
```

## Performance Considerations

### Cold Start
First forecast for an asset:
- API fetch: ~200-500ms
- ML training: ~50-100ms (30+ data points)
- Technical indicators: ~10-20ms
- **Total: ~300-700ms**

### Cached (Warm)
Repeat forecast within TTL:
- Cache lookup: <1ms
- **Total: <1ms**

### Optimization Tips
1. **Batch requests**: Use same asset ID to leverage cache
2. **Reuse SDK instance**: Shares cache across calls
3. **Adjust TTL**: Modify cache TTL for your use case
4. **Pre-warm cache**: Call `fetch_price_history` before forecasts
