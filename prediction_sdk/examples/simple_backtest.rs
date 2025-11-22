use chrono::{Duration, Utc};
use prediction_sdk::PredictionSdk;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🔄 Simple Walk-Forward Backtest");
    println!("================================\n");

    let sdk = PredictionSdk::new()?;
    let asset = "bitcoin";
    let currency = "usd";
    let lookback_days = 30;

    println!("📊 Fetching {} days of {} historical data...", lookback_days, asset);
    let now = Utc::now();
    let start = now - Duration::days(lookback_days);
    let full_history = sdk
        .fetch_price_history_range(asset, currency, start, now)
        .await?;

    if full_history.len() < 100 {
        eprintln!("❌ Insufficient data: need at least 100 points, got {}", full_history.len());
        return Ok(());
    }

    println!("✅ Fetched {} price points\n", full_history.len());
    println!("🔬 Running walk-forward backtest...\n");

    // We need at least 30 points for ML training, so start predictions from index 30
    let min_training_window = 30;
    let mut predictions = Vec::new();
    let mut errors = Vec::new();
    let mut correct_directions = 0;
    let mut total_predictions = 0;

    // Walk forward through the data
    for i in min_training_window..(full_history.len() - 1) {
        let training_data = &full_history[..i];
        let actual_next_price = full_history[i + 1].price;
        let current_price = full_history[i].price;

        // Make prediction using ML model
        match prediction_sdk::analysis::predict_next_price_ml(training_data) {
            Ok(ml_forecast) => {
                let predicted_price = ml_forecast.predicted_price;
                let error = (predicted_price - actual_next_price).abs();
                let relative_error = error / actual_next_price;

                // Check directional accuracy
                let predicted_direction = predicted_price > current_price;
                let actual_direction = actual_next_price > current_price;
                if predicted_direction == actual_direction {
                    correct_directions += 1;
                }

                predictions.push((
                    full_history[i + 1].timestamp,
                    predicted_price,
                    actual_next_price,
                    error,
                    relative_error,
                    ml_forecast.reliability,
                ));
                errors.push(error);
                total_predictions += 1;
            }
            Err(e) => {
                // Skip this prediction if ML model fails
                eprintln!("⚠️  Prediction failed at index {}: {}", i, e);
            }
        }
    }

    if predictions.is_empty() {
        eprintln!("❌ No successful predictions made");
        return Ok(());
    }

    // Calculate metrics
    let mae = errors.iter().sum::<f64>() / errors.len() as f64;
    let mse = errors.iter().map(|e| e.powi(2)).sum::<f64>() / errors.len() as f64;
    let rmse = mse.sqrt();
    let directional_accuracy = correct_directions as f64 / total_predictions as f64 * 100.0;

    // Calculate MAPE (Mean Absolute Percentage Error)
    let mape = predictions
        .iter()
        .map(|(_, _, _actual, _, relative_error, _)| relative_error * 100.0)
        .sum::<f64>()
        / predictions.len() as f64;

    // Calculate average reliability
    let avg_reliability = predictions
        .iter()
        .map(|(_, _, _, _, _, reliability)| *reliability as f64)
        .sum::<f64>()
        / predictions.len() as f64;

    // Print results
    println!("📈 Backtest Results");
    println!("==================");
    println!("Total Predictions:       {}", total_predictions);
    println!("Mean Absolute Error:     ${:.2}", mae);
    println!("Root Mean Squared Error: ${:.2}", rmse);
    println!("Mean Abs % Error (MAPE): {:.2}%", mape);
    println!("Directional Accuracy:    {:.1}%", directional_accuracy);
    println!("Average ML Reliability:  {:.1}%", avg_reliability * 100.0);

    // Show sample predictions
    println!("\n📋 Sample Predictions (Last 10)");
    println!("=====================================");
    println!("{:<20} {:>12} {:>12} {:>10}", "Timestamp", "Predicted", "Actual", "Error");
    println!("{}", "=".repeat(60));

    let sample_start = predictions.len().saturating_sub(10);
    for (timestamp, predicted, actual, error, _, _) in &predictions[sample_start..] {
        println!(
            "{:<20} ${:>11.2} ${:>11.2} ${:>9.2}",
            timestamp.format("%Y-%m-%d %H:%M"),
            predicted,
            actual,
            error
        );
    }

    // Calculate and show confidence-stratified accuracy
    println!("\n📊 Accuracy by Confidence Level");
    println!("================================");
    
    let high_confidence = predictions.iter().filter(|(_, _, _, _, _, rel)| *rel >= 0.7).collect::<Vec<_>>();
    let medium_confidence = predictions.iter().filter(|(_, _, _, _, _, rel)| *rel >= 0.4 && *rel < 0.7).collect::<Vec<_>>();
    let low_confidence = predictions.iter().filter(|(_, _, _, _, _, rel)| *rel < 0.4).collect::<Vec<_>>();

    if !high_confidence.is_empty() {
        let high_mae = high_confidence.iter().map(|(_, _, _, error, _, _)| error).sum::<f64>() / high_confidence.len() as f64;
        println!("High (≥70%):   {} predictions, MAE: ${:.2}", high_confidence.len(), high_mae);
    }
    
    if !medium_confidence.is_empty() {
        let med_mae = medium_confidence.iter().map(|(_, _, _, error, _, _)| error).sum::<f64>() / medium_confidence.len() as f64;
        println!("Medium (40-70%): {} predictions, MAE: ${:.2}", medium_confidence.len(), med_mae);
    }
    
    if !low_confidence.is_empty() {
        let low_mae = low_confidence.iter().map(|(_, _, _, error, _, _)| error).sum::<f64>() / low_confidence.len() as f64;
        println!("Low (<40%):    {} predictions, MAE: ${:.2}", low_confidence.len(), low_mae);
    }

    println!("\n✅ Backtest complete!");
    Ok(())
}
