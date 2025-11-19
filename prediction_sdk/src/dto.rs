use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ShortForecastHorizon {
    FifteenMinutes,
    OneHour,
    FourHours,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForecastDecomposition {
    pub trend: f64,
    pub momentum: f64,
    pub noise: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SentimentSnapshot {
    pub news_score: f64,
    pub social_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ForecastHorizon {
    Short(ShortForecastHorizon),
    Long(LongForecastHorizon),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ForecastResult {
    Short(ShortForecastResult),
    Long(LongForecastResult),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForecastRequest {
    pub asset_id: String,
    pub vs_currency: String,
    pub horizon: ForecastHorizon,
    pub sentiment: Option<SentimentSnapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShortForecastResult {
    pub horizon: ShortForecastHorizon,
    pub expected_price: f64,
    pub confidence: f32,
    pub decomposition: ForecastDecomposition,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LongForecastResult {
    pub horizon: LongForecastHorizon,
    pub mean_price: f64,
    pub percentile_10: f64,
    pub percentile_90: f64,
    pub confidence: f32,
}

#[derive(Debug, Error)]
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
