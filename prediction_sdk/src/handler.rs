use crate::{
    LongForecastHorizon,
    LongForecastResult,
    PredictionError,
    PredictionSdk,
    SentimentSnapshot,
    ShortForecastHorizon,
    ShortForecastResult,
};

pub async fn run_prediction_handler(
    asset_id: &str,
    vs_currency: &str,
) -> Result<(ShortForecastResult, LongForecastResult), PredictionError> {
    let sdk = PredictionSdk::new()?;
    let history = sdk.fetch_price_history(asset_id, vs_currency, 30).await?;
    let sentiment = SentimentSnapshot {
        news_score: 0.0,
        social_score: 0.0,
    };
    let short = sdk
        .run_short_forecast(&history, ShortForecastHorizon::OneHour, Some(sentiment.clone()))
        .await?;
    let long = sdk
        .run_long_forecast(
            &history,
            LongForecastHorizon::OneWeek,
            Some(sentiment),
        )
        .await?;

    Ok((short, long))
}
