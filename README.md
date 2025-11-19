# Inferenco Predictions SDK

Inferenco Predictions SDK is a Rust crate that fetches market data from CoinGecko and applies a blend of statistical signals (moving averages, volatility, Monte Carlo paths, and sentiment weighting) to produce short- and long-horizon cryptocurrency forecasts. The library is designed to be embedded into MCP (Model Context Protocol) tools, backend services, or research notebooks that need deterministic, fully auditable predictions.

## Project goals

- **Deterministic research tooling** – keep all forecasting logic inside a single Rust crate so MCP tools and backend services run the same calculations.
- **Composable architecture** – split DTOs, helpers, handlers, and implementations into focused modules so integrators can mix the SDK with their own orchestrators.
- **Practical forecasts** – expose both short-horizon signals (minutes to hours) and longer simulations (days to multi-year horizons) that combine price history, Monte Carlo sampling, and optional sentiment snapshots.
- **Operational safety** – lean on `reqwest` + `rustls`, typed errors via `thiserror`, and careful normalization to avoid panics in production code.

## Architecture overview

```
prediction_sdk/
├── src/
│   ├── dto.rs            # Data structs (price points, forecast DTOs, error enums)
│   ├── helpers.rs        # Local helpers (moving averages, Monte Carlo, sentiment math)
│   ├── handler.rs        # Ready-to-run handler that wires together SDK calls
│   ├── impl.rs           # `PredictionSdk` implementation and HTTP integration
│   ├── mod.rs            # Module exports (pub use of DTOs, handler, SDK)
│   └── lib.rs            # Crate entry point that re-exports the module root
```

**Pipelines**

1. **Market data ingestion** – `PredictionSdk::fetch_price_history` hits `https://api.coingecko.com/api/v3/coins/{id}/market_chart`, deserializes prices + volumes, and converts timestamps with `chrono`.
2. **Short-horizon signal** – `run_short_forecast` slices a rolling window, computes moving average + volatility, optionally weights the result with `SentimentSnapshot`, and reports `ForecastDecomposition` plus normalized confidence.
3. **Long-horizon simulation** – `run_long_forecast` derives the requested duration, executes Monte Carlo paths via `helpers::run_monte_carlo`, aggregates percentiles and mean, then applies optional sentiment weighting.
4. **Handler orchestration** – `handler::run_prediction_handler` demonstrates how to wire everything together: instantiate the SDK, fetch horizon-specific history (30-day lookback for short windows, explicit ranges for long ones), run the matching forecast, and return a single serialized result.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Rust toolchain | Rust **1.77+** with the 2024 edition components enabled (`rustup default stable` or newer nightly). |
| Cargo        | Installed automatically with Rust (`rustup`). |
| System libs  | No native OpenSSL dependency – `reqwest` is compiled with `rustls-tls` for easy builds. |

The crate depends on:

- `reqwest` (HTTP client, JSON + rustls)
- `tokio` (async runtime)
- `serde` / `serde_json` (serialization)
- `chrono` (timestamp parsing)
- `rand` + `statrs` (Monte Carlo + statistics)
- `thiserror` (error handling)

Install everything with `rustup component add rustfmt clippy` if you plan to lint locally.

## Setup & local development

```bash
# 1. Fetch the sources
$ git clone https://github.com/your-org/Inferenco-predictions-SDK.git
$ cd Inferenco-predictions-SDK/prediction_sdk

# 2. Build the crate
$ cargo build

# 3. Run the test suite (unit + doc tests)
$ cargo test
```

Because the SDK downloads live data during integration tests, keep an eye on CoinGecko rate limits while running `cargo test` repeatedly.

## Configuration (CoinGecko access)

- **API base URL** – `PredictionSdk::new()` defaults to the public `https://api.coingecko.com/api/v3` endpoint. To point to a proxy or authenticated mirror, construct your own `reqwest::Client` and call `PredictionSdk::with_client(client, Some(custom_url))`.
- **API keys** – CoinGecko’s community tier does **not** require an API key, and the current SDK does not add authentication headers automatically. If you have a paid CoinGecko plan, create a client with the appropriate default headers and pass it to `with_client` so every request includes your token.
- **Environment variables** – there are no mandatory env vars today. Downstream apps often export `COINGECKO_API_KEY` or `COINGECKO_BASE_URL` and read them before instantiating the SDK; the handler example below shows how to do that.

## Usage example

```rust
use chrono::{Duration, Utc};
use prediction_sdk::{
    ForecastHorizon,
    ForecastResult,
    PredictionSdk,
    SentimentSnapshot,
    ShortForecastHorizon,
    LongForecastHorizon,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optionally read env vars if you run against a private CoinGecko mirror.
    let base_url = std::env::var("COINGECKO_BASE_URL").ok();
    let client = reqwest::Client::new();
    let sdk = match base_url {
        Some(url) => PredictionSdk::with_client(client, Some(url)),
        None => PredictionSdk::new()?,
    };

    // Fetch price history and sentiment inputs.
    let history = sdk.fetch_price_history("bitcoin", "usd", 30).await?;
    let sentiment = SentimentSnapshot {
        news_score: 0.15,
        social_score: 0.05,
    };

    // One-call short horizon forecast (1 hour) with history fetched internally.
    let short = sdk
        .forecast_with_fetch(
            "bitcoin",
            "usd",
            ForecastHorizon::Short(ShortForecastHorizon::OneHour),
            Some(sentiment.clone()),
        )
        .await?;

    // Manual fetch + dispatch for a long horizon forecast (6 months)
    let long_horizon = LongForecastHorizon::SixMonths;
    let range_end = Utc::now();
    let range_start = range_end - Duration::days(180);
    let long_history = sdk
        .fetch_price_history_range("bitcoin", "usd", range_start, range_end)
        .await?;
    let long = sdk
        .forecast(
            &long_history,
            ForecastHorizon::Long(long_horizon),
            Some(sentiment),
        )
        .await?;

    match short {
        ForecastResult::Short(result) => println!("Short forecast: {:?}", result),
        ForecastResult::Long(_) => unreachable!("short forecast returned long result"),
    }

    match long {
        ForecastResult::Long(result) => println!("Long forecast: {:?}", result),
        ForecastResult::Short(_) => unreachable!("long forecast returned short result"),
    }
    Ok(())
}
```

See `handler::run_prediction_handler` for a ready-made orchestration function that bundles the above steps for typical MCP tools or CLI entry points. It accepts a `ForecastRequest` containing the asset ID, `vs_currency`, target horizon, and optional sentiment snapshot, fetches the appropriate price history (using the range API for long horizons), and dispatches to the correct forecast method before returning JSON.

### Selecting between `days` and `range`

- **`fetch_price_history` (days-based)** – use this for short to medium lookbacks (e.g., 7–90 days) where a rolling number of days is sufficient and you prefer a concise request.
- **`fetch_price_history_range` (from/to)** – use this for long horizons or precise research windows. Pass explicit `DateTime<Utc>` bounds to make sure you capture exactly the lookback period required by your forecast horizon. For compatibility, `fetch_market_chart_range` proxies to this method.

### Choosing a long-horizon forecast

`LongForecastHorizon` now spans tactical and strategic windows so you can match the simulation to your use case:

| Variant | Approximate days | When to use |
|---------|------------------|-------------|
| `OneDay`, `ThreeDays`, `OneWeek` | 1–7 days | Intra-week risk, short squeezes, liquidation alerts. |
| `OneMonth` | 30 days | Monthly rebalancing, post-event digestion. |
| `ThreeMonths`, `SixMonths` | 90 / 180 days | Quarterly treasury planning or macro-driven theses. |
| `OneYear`, `FourYears` | ~360 / ~1,440 days | Long-term strategy, mining runway modeling, and protocol treasury projections. |

Internally the SDK scales the Monte Carlo simulation count as horizons expand so that week-long runs stay highly granular while multi-year scenarios remain computationally practical.

## MCP integration notes

- **Tooling contract** – DTOs such as `ShortForecastResult`, `LongForecastResult`, and `ForecastDecomposition` live in `dto.rs`, making it trivial to serialize them as JSON payloads over MCP.
- **Handler hook** – Expose `run_prediction_handler(request: ForecastRequest) -> Result<String, PredictionError>` as an MCP command handler to keep the host lightweight; the handler owns SDK initialization, data fetching, and horizon-specific forecast calls.
- **Async runtime** – MCP servers written in Rust can reuse the `tokio` runtime already required by the SDK. Ensure your MCP dispatcher awaits the returned futures.
- **Error mapping** – Convert `PredictionError` into the MCP error surface (e.g., `ToolError` or JSON-RPC error) so clients receive human-readable failure reasons (`Network`, `Serialization`, etc.).

With these guidelines, the Inferenco Predictions SDK can power MCP tools, backend cron jobs, or research workflows with the same deterministic forecasting logic.

**Example MCP payload**

```json
{
  "asset_id": "bitcoin",
  "vs_currency": "usd",
  "horizon": { "type": "short", "value": "one_hour" },
  "sentiment": { "news_score": 0.1, "social_score": -0.05 }
}
```

When `sentiment` is omitted, the handler defaults to a neutral snapshot (`0.0` news and social scores) before invoking the short- or long-horizon forecast and returning the serialized JSON result.
