use chrono::{Duration, Utc};

use crate::{
    helpers,
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
    let long_horizon = LongForecastHorizon::OneWeek;
    let now = Utc::now();
    let lookback_days = helpers::long_horizon_days(long_horizon);
    let start = now - Duration::days(i64::from(lookback_days));
    let long_history =
        sdk.fetch_market_chart_range(asset_id, vs_currency, start, now).await?;
    let sentiment = SentimentSnapshot {
        news_score: 0.0,
        social_score: 0.0,
    };
    let short = sdk
        .run_short_forecast(&history, ShortForecastHorizon::OneHour, Some(sentiment.clone()))
        .await?;
    let long = sdk
        .run_long_forecast(
            &long_history,
            long_horizon,
            Some(sentiment),
        )
        .await?;

    Ok((short, long))
}
