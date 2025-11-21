use prediction_sdk::{
    ForecastHorizon, ForecastRequest, LongForecastHorizon, run_prediction_handler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Prediction SDK for Ethereum (1y)...");

    let request = ForecastRequest {
        asset_id: "ethereum".to_string(),
        vs_currency: "usd".to_string(),
        horizon: ForecastHorizon::Long(LongForecastHorizon::OneYear),
        sentiment: None,
        chart: true,
    };

    println!(
        "📊 Fetching data and forecasting for {}...",
        request.asset_id
    );

    let json_response = run_prediction_handler(request).await?;

    // Pretty print the JSON so the user can see the data
    let parsed: serde_json::Value = serde_json::from_str(&json_response)?;
    println!("{}", serde_json::to_string_pretty(&parsed)?);

    Ok(())
}
