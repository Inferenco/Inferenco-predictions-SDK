# Inferenco Predictions SDK

A Rust library for cryptocurrency price forecasting that combines statistical methods, technical analysis, and machine learning.

## Features

- **Hybrid Forecasting Engine**: Ensemble of statistical models, ML (SVR), and technical indicators
- **Multi-Horizon Forecasts**: From 15 minutes to 4 years
- **Two-Tier Caching**: Smart TTL-based caching for API responses and forecasts
- **Rate Limiting**: Built-in token bucket + exponential backoff for CoinGecko API
- **Technical Analysis**: RSI, MACD, Bollinger Bands
- **Local AI**: On-the-fly SmartCore SVR training (no external models needed)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
prediction_sdk = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```bash
cd prediction_sdk
cargo build
cargo test
```

## Usage

```rust
use prediction_sdk::{PredictionSdk, ForecastHorizon, ShortForecastHorizon, SentimentSnapshot};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdk = PredictionSdk::new()?;
    
    let sentiment = SentimentSnapshot {
        news_score: 0.65,
        social_score: 0.80,
    };
    
    let result = sdk.forecast_with_fetch(
        "bitcoin",
        "usd",
        ForecastHorizon::Short(ShortForecastHorizon::OneHour),
        Some(sentiment),
    ).await?;
    
    println!("Forecast: {:?}", result);
    Ok(())
}
```

### Chart responses

When `chart` is enabled on a request, the SDK augments the forecast with
historical candles derived from CoinGecko OHLC data (or locally aggregated from
the fetched price series). Each candle includes an ISO-8601 timestamp marking
the bucket start, open/high/low/close prices, and an optional aggregated volume
when the upstream endpoint supplies it.

```json
{
  "forecast": { "type": "short", "value": { /* ... */ } },
  "chart": {
    "history": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 42000.0,
        "high": 42500.0,
        "low": 41800.0,
        "close": 42350.0,
        "volume": 1823.4
      }
    ],
    "projection": [
      {"timestamp": "2024-01-02T00:00:00Z", "percentile_10": 40000.0, "mean": 43000.0, "percentile_90": 46000.0}
    ]
  }
}
```

## Run the Example

```bash
cargo run --example real_world_test
```

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed architecture, API reference, and integration guides.

## CI/CD

GitHub Actions runs:
- `cargo fmt --check`
- `cargo clippy`
- `cargo build`
- `cargo test` (unit + integration + doc tests)

## License

MIT
