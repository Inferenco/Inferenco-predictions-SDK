use prediction_sdk::{ForecastHorizon, ForecastResult, PredictionSdk, ShortForecastHorizon};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Backtesting Stohn Coin 1-Hour Forecasts");
    println!("==========================================\n");

    let sdk = PredictionSdk::new()?;
    let asset = "stohn-coin";
    let horizon = ForecastHorizon::Short(ShortForecastHorizon::OneHour);

    // Fetch 30 days of historical data for backtesting
    println!("📊 Fetching historical data...");
    let lookback_days = 30;
    let all_history = sdk.fetch_price_history(asset, "usd", lookback_days).await?;

    if all_history.len() < 100 {
        eprintln!("❌ Insufficient data for backtesting");
        return Ok(());
    }

    println!("✅ Loaded {} price points", all_history.len());
    println!(
        "📅 Data range: {} to {}\n",
        all_history.first().unwrap().timestamp,
        all_history.last().unwrap().timestamp
    );

    // Walk forward through the data
    let min_train_size = 50; // Minimum history for training
    let step_size = 24; // Test every 24 hours
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();
    let mut errors = Vec::new();

    println!("🔄 Running walk-forward backtest...\n");

    for idx in (min_train_size..all_history.len() - 1).step_by(step_size) {
        let train_data = &all_history[..idx];
        let actual_next = all_history[idx + 1].price;

        match sdk.forecast(train_data, horizon.clone(), None).await {
            Ok(ForecastResult::Short(forecast)) => {
                let predicted = forecast.expected_price;
                let error = (predicted - actual_next).abs();
                let direction_correct = (predicted > train_data.last().unwrap().price)
                    == (actual_next > train_data.last().unwrap().price);

                predictions.push(predicted);
                actuals.push(actual_next);
                errors.push(error);

                println!(
                    "  📅 {} | Predicted: ${:.4} | Actual: ${:.4} | Error: ${:.4} | Direction: {}",
                    all_history[idx].timestamp.format("%Y-%m-%d %H:%M"),
                    predicted,
                    actual_next,
                    error,
                    if direction_correct { "✓" } else { "✗" }
                );
            }
            Err(e) => {
                eprintln!("  ⚠️  Forecast failed at index {}: {}", idx, e);
            }
            _ => {}
        }
    }

    // Calculate metrics
    if !errors.is_empty() {
        println!("\n📊 Backtest Results");
        println!("===================");

        let mae = errors.iter().sum::<f64>() / errors.len() as f64;
        let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();

        let mape = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(pred, actual)| ((pred - actual).abs() / actual.abs()) * 100.0)
            .sum::<f64>()
            / predictions.len() as f64;

        let direction_correct = predictions
            .iter()
            .zip(actuals.iter())
            .enumerate()
            .filter(|(i, (pred, actual))| {
                if *i == 0 {
                    return false;
                }
                let prev_actual = actuals[i - 1];
                (**pred > prev_actual) == (**actual > prev_actual)
            })
            .count();

        let direction_accuracy = if predictions.len() > 1 {
            (direction_correct as f64 / (predictions.len() - 1) as f64) * 100.0
        } else {
            0.0
        };

        println!("Total Predictions: {}", predictions.len());
        println!("MAE (Mean Absolute Error): ${:.4}", mae);
        println!("RMSE (Root Mean Squared Error): ${:.4}", rmse);
        println!("MAPE (Mean Absolute Percentage Error): {:.2}%", mape);
        println!("Directional Accuracy: {:.1}%", direction_accuracy);

        println!("\n✅ Backtest complete!");
    } else {
        println!("\n❌ No successful predictions");
    }

    Ok(())
}
