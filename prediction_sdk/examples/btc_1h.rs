use prediction_sdk::{ForecastHorizon, ForecastResult, PredictionSdk, ShortForecastHorizon};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Prediction SDK for Bitcoin (1h)...");
    let sdk = PredictionSdk::new()?;

    let asset = "bitcoin";
    let horizon = ForecastHorizon::Short(ShortForecastHorizon::OneHour);

    println!("📊 Fetching data and forecasting for {}...", asset);

    match sdk.forecast_with_fetch(asset, "usd", horizon, None).await {
        Ok(result) => {
            if let ForecastResult::Short(res) = result {
                println!("\n--- ⏱️  1 Hour Bitcoin Forecast ---");
                println!("      💰 Expected Price: ${:.2}", res.expected_price);
                if let Some((lower, upper)) = res.ml_price_interval {
                    println!("      📉 Bearish (10th): ${:.2}", lower);
                    println!("      📈 Bullish (90th): ${:.2}", upper);
                }
                println!("      🎯 Confidence: {:.1}%", res.confidence * 100.0);
                if let Some(ml_price) = res.ml_prediction {
                    println!("      🤖 AI Prediction: ${:.2}", ml_price);
                }
            }
        }
        Err(e) => eprintln!("❌ Forecast failed: {}", e),
    }

    Ok(())
}
