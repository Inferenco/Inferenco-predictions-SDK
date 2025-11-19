use prediction_sdk::{
    ForecastHorizon, ForecastResult, PredictionSdk, SentimentSnapshot, ShortForecastHorizon,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 Initializing Prediction SDK...");
    let sdk = PredictionSdk::new()?;

    // 1. Define parameters
    let asset = "bitcoin";
    let currency = "usd";
    let sentiment = SentimentSnapshot {
        news_score: 0.65,  // Slightly bullish news
        social_score: 0.8, // Very bullish social sentiment
    };

    println!(
        "\n📊 Fetching data and forecasting for {} in {}...",
        asset, currency
    );
    println!(
        "   Sentiment: News={:.2}, Social={:.2}",
        sentiment.news_score, sentiment.social_score
    );

    // 2. Run Short-Term Forecasts
    println!("\n--- ⏱️  Short-Term Forecasts ---");
    let short_horizons = vec![
        ("15 Minutes", ShortForecastHorizon::FifteenMinutes),
        ("1 Hour", ShortForecastHorizon::OneHour),
        ("4 Hours", ShortForecastHorizon::FourHours),
    ];

    for (label, horizon) in short_horizons {
        println!("\n   👉 {}:", label);
        let short_result = sdk
            .forecast_with_fetch(
                asset,
                currency,
                ForecastHorizon::Short(horizon),
                Some(sentiment.clone()),
            )
            .await?;

        if let ForecastResult::Short(res) = short_result {
            println!("      💰 Expected Price: ${:.2}", res.expected_price);
            println!("      🎯 Confidence: {:.1}%", res.confidence * 100.0);

            if let Some(ml_price) = res.ml_prediction {
                println!("      🤖 AI Prediction: ${:.2}", ml_price);
            }
        }
    }

    // 3. Run Long-Term Forecasts
    println!("\n--- 🗓️  Long-Term Forecasts ---");
    let long_horizons = vec![
        ("1 Month", prediction_sdk::LongForecastHorizon::OneMonth),
        ("3 Months", prediction_sdk::LongForecastHorizon::ThreeMonths),
        ("1 Year", prediction_sdk::LongForecastHorizon::OneYear),
    ];

    for (label, horizon) in long_horizons {
        println!("\n   👉 {}:", label);
        let long_result = sdk
            .forecast_with_fetch(
                asset,
                currency,
                ForecastHorizon::Long(horizon),
                Some(sentiment.clone()),
            )
            .await?;

        if let ForecastResult::Long(res) = long_result {
            println!("      💰 Mean Price: ${:.2}", res.mean_price);
            println!("      📉 Bearish (10th): ${:.2}", res.percentile_10);
            println!("      📈 Bullish (90th): ${:.2}", res.percentile_90);
            println!("      🎯 Confidence: {:.1}%", res.confidence * 100.0);
        }
    }

    Ok(())
}
