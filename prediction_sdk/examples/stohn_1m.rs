use prediction_sdk::{ForecastHorizon, ForecastResult, LongForecastHorizon, PredictionSdk};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Prediction SDK for Stohn Coin (1m)...");
    let sdk = PredictionSdk::new()?;

    let asset = "stohn-coin";
    let horizon = ForecastHorizon::Long(LongForecastHorizon::OneMonth);

    println!("📊 Fetching data and forecasting for {}...", asset);

    match sdk.forecast_with_fetch(asset, "usd", horizon, None).await {
        Ok(result) => {
            if let ForecastResult::Long(res) = result {
                println!("\n--- 🗓️  1 Month Forecast ---");
                println!("      💰 Expected Price: ${:.4}", res.mean_price);
                println!("      📉 Bearish (10th): ${:.4}", res.percentile_10);
                println!("      📈 Bullish (90th): ${:.4}", res.percentile_90);
                println!("      🎯 Confidence: {:.1}%", res.confidence * 100.0);
                if let Some(ml_price) = res.ml_prediction {
                    println!("      🤖 AI Prediction: ${:.4}", ml_price);
                }
            }
        }
        Err(e) => eprintln!("❌ Forecast failed: {}", e),
    }

    Ok(())
}
