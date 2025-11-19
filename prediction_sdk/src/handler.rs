use chrono::{Duration, Utc};

use crate::{
    ForecastHorizon, ForecastRequest, ForecastResult, PredictionError, PredictionSdk,
    SentimentSnapshot, helpers,
};

const SHORT_FORECAST_LOOKBACK_DAYS: u32 = 30;

/// Execute a forecast based on a [`ForecastRequest`], returning serialized JSON.
///
/// This helper is intended for MCP or HTTP entry points where callers send a
/// single payload describing the asset, quote currency, desired horizon, and
/// optional sentiment. The handler fetches the correct lookback window for the
/// requested horizon (range-based for long forecasts, rolling days for short
/// forecasts), dispatches to the appropriate SDK method, and returns a JSON
/// string containing a [`ForecastResult`].
///
/// # Examples
///
/// ```no_run
/// use prediction_sdk::{
///     ForecastHorizon, ForecastRequest, SentimentSnapshot, ShortForecastHorizon,
/// };
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), prediction_sdk::PredictionError> {
/// let request = ForecastRequest {
///     asset_id: "bitcoin".to_string(),
///     vs_currency: "usd".to_string(),
///     horizon: ForecastHorizon::Short(ShortForecastHorizon::OneHour),
///     sentiment: Some(SentimentSnapshot {
///         news_score: 0.1,
///         social_score: 0.05,
///     }),
/// };
///
/// let json = prediction_sdk::run_prediction_handler(request).await?;
/// println!("{json}");
/// # Ok(())
/// # }
/// ```
pub async fn run_prediction_handler(request: ForecastRequest) -> Result<String, PredictionError> {
    let sdk = PredictionSdk::new()?;
    let sentiment = request.sentiment.unwrap_or(SentimentSnapshot {
        news_score: 0.0,
        social_score: 0.0,
    });

    let forecast = match request.horizon {
        ForecastHorizon::Short(horizon) => {
            let history = sdk
                .fetch_price_history(
                    &request.asset_id,
                    &request.vs_currency,
                    SHORT_FORECAST_LOOKBACK_DAYS,
                )
                .await?;

            sdk.run_short_forecast(&history, horizon, Some(sentiment.clone()))
                .await
                .map(ForecastResult::Short)
        }
        ForecastHorizon::Long(horizon) => {
            let now = Utc::now();
            let lookback_days = helpers::long_horizon_days(horizon);
            let start = now - Duration::days(i64::from(lookback_days));
            let history = sdk
                .fetch_price_history_range(&request.asset_id, &request.vs_currency, start, now)
                .await?;

            sdk.run_long_forecast(&history, horizon, Some(sentiment.clone()))
                .await
                .map(ForecastResult::Long)
        }
    }?;

    serde_json::to_string(&forecast).map_err(|err| PredictionError::Serialization(err.to_string()))
}
