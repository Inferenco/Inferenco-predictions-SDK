use prediction_sdk::{ForecastHorizon, LongForecastHorizon, PredictionSdk};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdk = PredictionSdk::new()?;

    println!("🚀 Initializing Prediction SDK for Aptos (4y)...");

    // Run a 4-year forecast
    // This normally requires 4 years of history.
    // With the fix, it should cap at 1 year (365 days) and succeed.
    println!("📊 Fetching data and forecasting for aptos...");
    let result = sdk
        .forecast_with_fetch(
            "aptos",
            "usd",
            ForecastHorizon::Long(LongForecastHorizon::FourYears),
            None,
        )
        .await;

    match result {
        Ok(forecast) => {
            if let prediction_sdk::ForecastResult::Long(long_forecast) = forecast {
                println!("\n--- 🗓️  4 Year Forecast ---");
                println!("      💰 Expected Price: ${:.4}", long_forecast.mean_price);
                println!(
                    "      📉 Bearish (10th): ${:.4}",
                    long_forecast.percentile_10
                );
                println!(
                    "      📈 Bullish (90th): ${:.4}",
                    long_forecast.percentile_90
                );
                println!(
                    "      🎯 Confidence: {:.1}%",
                    long_forecast.confidence * 100.0
                );
            }
        }
        Err(e) => {
            eprintln!("❌ Forecast failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
