use prediction_sdk::{ForecastHorizon, LongForecastHorizon, PredictionSdk};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 Initializing Prediction SDK...");
    let sdk = PredictionSdk::new()?;

    let asset_id = "bitcoin";
    println!("📊 Fetching live sentiment for {}...", asset_id);

    // 1. Fetch Live Sentiment
    let sentiment = sdk.fetch_live_sentiment(asset_id).await?;

    println!("\n--- 🧠 Live Sentiment Snapshot ---");
    println!(
        "   📰 News Score (Fear & Greed): {:.2} (Mapped from 0-100 to -1.0-1.0)",
        sentiment.news_score
    );
    println!(
        "   🗣️ Social Score (Community):  {:.2} (Mapped from 0-100 to -1.0-1.0)",
        sentiment.social_score
    );

    // 2. Run Forecast with Sentiment
    println!("\n🔮 Running 1-Month Forecast with Sentiment Bias...");
    let result = sdk
        .forecast_with_fetch(
            asset_id,
            "usd",
            ForecastHorizon::Long(LongForecastHorizon::OneMonth),
            Some(sentiment),
        )
        .await?;

    if let prediction_sdk::ForecastResult::Long(forecast) = result {
        println!("\n--- 🗓️  1 Month Forecast Result ---");
        println!("      💰 Mean Price: ${:.2}", forecast.mean_price);
        println!("      📉 Bearish:    ${:.2}", forecast.percentile_10);
        println!("      📈 Bullish:    ${:.2}", forecast.percentile_90);

        if let Some(echoed_sentiment) = forecast.sentiment {
            println!("\n      🧠 Echoed Sentiment:");
            println!("         - News:   {:.2}", echoed_sentiment.news_score);
            println!("         - Social: {:.2}", echoed_sentiment.social_score);
        }

        if let Some(paths) = forecast.sample_paths {
            println!("\n      〰️  Sample Paths Generated: {}", paths.len());
            for path in paths {
                println!(
                    "         - {}: Ends at ${:.2}",
                    path.label,
                    path.points.last().unwrap_or(&0.0)
                );
            }
        }
    }

    Ok(())
}
