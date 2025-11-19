use chrono::{Duration, Utc};

use crate::{
    helpers,
    ForecastHorizon,
    ForecastRequest,
    PredictionError,
    PredictionSdk,
    SentimentSnapshot,
};

pub async fn run_prediction_handler(
    request: ForecastRequest,
) -> Result<String, PredictionError> {
    let sdk = PredictionSdk::new()?;
    let sentiment = request.sentiment.unwrap_or(SentimentSnapshot {
        news_score: 0.0,
        social_score: 0.0,
    });

    match request.horizon {
        ForecastHorizon::Short(short_horizon) => {
            let history = sdk
                .fetch_price_history(&request.token_id, &request.vs_currency, 30)
                .await?;
            let forecast = sdk
                .run_short_forecast(&history, short_horizon, Some(sentiment))
                .await?;
            serde_json::to_string(&forecast)
                .map_err(|err| PredictionError::Serialization(err.to_string()))
        }
        ForecastHorizon::Long(long_horizon) => {
            let now = Utc::now();
            let lookback_days = helpers::long_horizon_days(long_horizon);
            let start = now - Duration::days(i64::from(lookback_days));
            let history = sdk
                .fetch_market_chart_range(&request.token_id, &request.vs_currency, start, now)
                .await?;
            let forecast = sdk
                .run_long_forecast(
                    &history,
                    long_horizon,
                    Some(sentiment),
                )
                .await?;
            serde_json::to_string(&forecast)
                .map_err(|err| PredictionError::Serialization(err.to_string()))
        }
    }
}
