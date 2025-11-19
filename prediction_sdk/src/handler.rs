use chrono::{Duration, Utc};

use crate::{
    helpers,
    ForecastHorizon,
    ForecastRequest,
    ForecastResult,
    PredictionError,
    PredictionSdk,
    SentimentSnapshot,
};

const SHORT_FORECAST_LOOKBACK_DAYS: u32 = 30;

pub async fn run_prediction_handler(
    request: ForecastRequest,
) -> Result<String, PredictionError> {
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

            sdk
                .run_short_forecast(&history, horizon, Some(sentiment.clone()))
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

    serde_json::to_string(&forecast)
        .map_err(|err| PredictionError::Serialization(err.to_string()))
}
