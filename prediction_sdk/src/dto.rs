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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Candlestick data derived from CoinGecko endpoints or locally aggregated.
///
/// * `timestamp` marks the start of the bucket (in UTC) represented by the
///   OHLC values.
/// * `open`, `high`, `low`, and `close` capture the price movement within the
///   bucket.
/// * `volume` aggregates the traded volume when the upstream series provides
///   it; otherwise it is omitted.
pub struct ChartCandle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
/// Short forecast horizons used for intraday predictions.
///
/// Serialized with `snake_case` strings (e.g., `"one_hour"`).
pub enum ShortForecastHorizon {
    FifteenMinutes,
    OneHour,
    FourHours,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
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
    TwoYears,
    ThreeYears,
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
pub struct TechnicalSignals {
    pub rsi: f64,
    pub macd_divergence: f64,
    pub bollinger_width: f64,
    pub trend_strength: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MlForecast {
    pub predicted_price: f64,
    pub predicted_return: f64,
    pub lower_return: f64,
    pub upper_return: f64,
    pub calibration_score: f32,
    pub target_coverage: f64,
    pub observed_coverage: f64,
    pub interval_width: f64,
    pub pinball_loss: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IntervalCalibration {
    pub target_coverage: f64,
    pub observed_coverage: f64,
    pub interval_width: f64,
    pub price_interval_width: f64,
    pub pinball_loss: f64,
    pub calibration_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CovariatePoint {
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub macro_covariates: Vec<f64>,
    #[serde(default)]
    pub onchain_covariates: Vec<f64>,
    #[serde(default)]
    pub sentiment_covariates: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MlModelKind {
    MixLinear,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MlModelConfig {
    pub model: MlModelKind,
    pub patch_length: usize,
    pub mixture_components: usize,
    pub learning_rate: f64,
    pub validation_window: usize,
    pub validation_stride: usize,
}

impl Default for MlModelConfig {
    fn default() -> Self {
        Self {
            model: MlModelKind::MixLinear,
            patch_length: 8,
            mixture_components: 3,
            learning_rate: 0.01,
            validation_window: 6,
            validation_stride: 3,
        }
    }
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
    #[serde(default)]
    pub chart: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForecastResponse {
    pub forecast: ForecastResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chart: Option<ForecastChart>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SimulationStepSample {
    pub day: u32,
    pub mean: f64,
    pub percentile_10: f64,
    pub percentile_90: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MonteCarloRun {
    pub final_prices: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_samples: Option<Vec<SimulationStepSample>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub technical_signals: Option<TechnicalSignals>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_prediction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_return_bounds: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_price_interval: Option<(f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_interval_calibration: Option<IntervalCalibration>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SamplePath {
    pub label: String,
    pub points: Vec<f64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub technical_signals: Option<TechnicalSignals>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_prediction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_paths: Option<Vec<SamplePath>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForecastBandPoint {
    pub timestamp: DateTime<Utc>,
    pub percentile_10: f64,
    pub mean: f64,
    pub percentile_90: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForecastChart {
    pub history: Vec<ChartCandle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection: Option<Vec<ForecastBandPoint>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MonteCarloBenchmark {
    pub horizon_days: u32,
    pub constant_mean: f64,
    pub constant_percentile_10: f64,
    pub constant_percentile_90: f64,
    pub regime_mean: f64,
    pub regime_percentile_10: f64,
    pub regime_percentile_90: f64,
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
