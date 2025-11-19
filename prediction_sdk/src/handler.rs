use crate::{
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

    sdk.forecast_with_fetch(
        &request.token_id,
        &request.vs_currency,
        request.horizon,
        Some(sentiment),
    )
    .await
    .and_then(|forecast| {
        serde_json::to_string(&forecast)
            .map_err(|err| PredictionError::Serialization(err.to_string()))
    })
}
