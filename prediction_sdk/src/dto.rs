use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Time-series market observation used for forecasting.
///
/// * `timestamp` is expressed in UTC seconds with millisecond precision and
///   comes from the upstream CoinGecko market chart endpoint.
/// * `price` is the quoted asset price in the provided `vs_currency`.
/// * `volume` represents the traded volume within the upstream bucket, if
///   present. Some endpoints omit volume data, so the field is optional.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
/// Short forecast horizons used for intraday predictions.
///
/// Serialized with `snake_case` strings (e.g., `"one_hour"`).
pub enum ShortForecastHorizon {
    FifteenMinutes,
    OneHour,
    FourHours,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
/// Longer-term forecast horizons ranging from one day to four years.
///
/// Serialized with `snake_case` strings (e.g., `"one_month"`).
pub enum LongForecastHorizon {
    OneDay,
    ThreeDays,
    OneWeek,
    OneMonth,
    ThreeMonths,
    SixMonths,
    OneYear,
    FourYears,
}

/// Breakdown of a short-term forecast signal.
///
/// Values are normalized contributions derived from the input series and do not
/// represent absolute price levels.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForecastDecomposition {
    pub trend: f64,
    pub momentum: f64,
    pub noise: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Sentiment snapshot applied as an optional modifier to forecasts.
///
/// Scores typically fall within `[-1.0, 1.0]` and are assumed to be normalized
/// upstream. They are treated as dimensionless multipliers.
pub struct SentimentSnapshot {
    pub news_score: f64,
    pub social_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
/// Wrapper for selecting either a short or long forecast horizon.
///
/// Serialized as an externally tagged enum with `type`/`value` keys to match
/// the API shape, for example:
///
/// ```json
/// { "type": "short", "value": "one_hour" }
/// ```
pub enum ForecastHorizon {
    Short(ShortForecastHorizon),
    Long(LongForecastHorizon),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
/// Forecast output paired with the horizon variant.
///
/// Uses the same externally tagged format as [`ForecastHorizon`].
pub enum ForecastResult {
    Short(ShortForecastResult),
    Long(LongForecastResult),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Request payload for generating forecasts.
///
/// * `asset_id` corresponds to the CoinGecko asset identifier (e.g.,
///   `"bitcoin"`).
/// * `vs_currency` is the quote currency for pricing (e.g., `"usd"`).
/// * `horizon` selects the desired forecast length.
/// * `sentiment` optionally adjusts the forecast using the provided snapshot.
pub struct ForecastRequest {
    pub asset_id: String,
    pub vs_currency: String,
    pub horizon: ForecastHorizon,
    pub sentiment: Option<SentimentSnapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Result of a short-horizon forecast.
///
/// * `expected_price` is the predicted point estimate in the requested
///   `vs_currency`.
/// * `confidence` is a normalized score in `[0.0, 1.0]` derived from volatility
///   and signal noise.
/// * `decomposition` exposes the relative contribution of trend, momentum, and
///   noise used to build the forecast.
pub struct ShortForecastResult {
    pub horizon: ShortForecastHorizon,
    pub expected_price: f64,
    pub confidence: f32,
    pub decomposition: ForecastDecomposition,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Result of a long-horizon forecast.
///
/// * `mean_price` is the Monte Carlo mean price in the requested
///   `vs_currency`.
/// * `percentile_10` and `percentile_90` provide a simple confidence band
///   around the mean.
/// * `confidence` is a normalized score in `[0.0, 1.0]` that decreases for
///   longer horizons.
pub struct LongForecastResult {
    pub horizon: LongForecastHorizon,
    pub mean_price: f64,
    pub percentile_10: f64,
    pub percentile_90: f64,
    pub confidence: f32,
}

#[derive(Debug, Error)]
/// Errors that can be returned by the SDK when fetching data or producing
/// forecasts.
pub enum PredictionError {
    #[error("network call failed: {0}")]
    Network(String),
    #[error("failed to deserialize response: {0}")]
    Serialization(String),
    #[error("time conversion failed")]
    TimeConversion,
    #[error("insufficient data for calculation")]
    InsufficientData,
}
