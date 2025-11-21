use prediction_sdk::{
    ForecastHorizon, ForecastRequest, ShortForecastHorizon, run_prediction_handler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Prediction SDK for Bitcoin (15m)...");

    let request = ForecastRequest {
        asset_id: "bitcoin".to_string(),
        vs_currency: "usd".to_string(),
        horizon: ForecastHorizon::Short(ShortForecastHorizon::FifteenMinutes),
        sentiment: None,
        chart: true,
    };

    println!(
        "📊 Fetching data and forecasting for {}...",
        request.asset_id
    );

    // We use the handler here to easily get the JSON response which includes the chart
    let json_response = run_prediction_handler(request).await?;

    // Pretty print the JSON so the user can see the data
    let parsed: serde_json::Value = serde_json::from_str(&json_response)?;
    println!("{}", serde_json::to_string_pretty(&parsed)?);

    Ok(())
}
