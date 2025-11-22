use prediction_sdk::{
    ForecastHorizon, ForecastRequest, ForecastResult, ShortForecastHorizon, run_prediction_handler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Prediction SDK for Bitcoin (15m, no chart)...");

    let request = ForecastRequest {
        asset_id: "bitcoin".to_string(),
        vs_currency: "usd".to_string(),
        horizon: ForecastHorizon::Short(ShortForecastHorizon::FifteenMinutes),
        sentiment: None,
        chart: false,
    };

    println!(
        "📊 Fetching data and forecasting for {}...",
        request.asset_id
    );

    // We use the handler here to easily get the JSON response
    let json_response = run_prediction_handler(request).await?;

    // Try to deserialize directly into ForecastResult to verify the structure
    // If chart was true, this would fail because the root object would be ForecastResponse
    let result: ForecastResult = serde_json::from_str(&json_response)?;

    println!("✅ Successfully deserialized ForecastResult (no chart wrapper).");

    if let ForecastResult::Short(res) = result {
        println!("\n--- ⏱️  15 Minute Forecast ---");
        println!("      💰 Expected Price: ${:.4}", res.expected_price);
        println!("      🎯 Confidence: {:.1}%", res.confidence * 100.0);
        if let Some(ml_price) = res.ml_prediction {
            println!("      🤖 AI Prediction: ${:.4}", ml_price);
        }
    } else {
        eprintln!("❌ Unexpected forecast type!");
        std::process::exit(1);
    }

    Ok(())
}
