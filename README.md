# Inferenco Predictions SDK

A Rust library for cryptocurrency price forecasting that combines statistical methods, technical analysis, and machine learning.

## Projects using this SDK

- [DeFiCalc](https://deficalc.io/) integrates the SDK for price-aware DeFi calculations.
- [Inferenco Nova](https://inferenco.com/app.html#nova) uses the SDK to power AI-driven blockchain conversations and tooling.

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

### Fetching Charts

To retrieve historical candles and (for long horizons) projection bands, set the `chart` flag to `true` in your request. This is best handled via the `run_prediction_handler` helper or by manually calling `fetch_chart_candles`.

```rust
use prediction_sdk::{ForecastRequest, ForecastHorizon, ShortForecastHorizon, run_prediction_handler};

// ... inside async fn
let request = ForecastRequest {
    asset_id: "bitcoin".to_string(),
    vs_currency: "usd".to_string(),
    horizon: ForecastHorizon::Short(ShortForecastHorizon::OneHour),
    sentiment: None,
    chart: true, // <--- Enable chart data
};

// Returns a JSON string containing both "forecast" and "chart" fields
let json_response = run_prediction_handler(request).await?;
println!("{}", json_response);
```

**Response Structure:**

```json
{
  "forecast": {
    "type": "short",
    "value": {
      "expected_price": 42350.0,
      "confidence": 0.85,
      // ... other forecast fields
    }
  },
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
      // ... more candles
    ],
    "projection": [
      // Only present for Long horizons
      {
        "timestamp": "2024-01-02T00:00:00Z", 
        "percentile_10": 40000.0, 
        "mean": 43000.0, 
        "percentile_90": 46000.0
      }
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
