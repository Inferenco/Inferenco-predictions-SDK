use chrono::{Duration, Utc};
use prediction_sdk::PredictionSdk;
use std::error::Error;

#[derive(Debug)]
struct IntervalValidation {
    timestamp: chrono::DateTime<chrono::Utc>,
    actual_price: f64,
    predicted_price: f64,
    lower_bound: f64,
    upper_bound: f64,
    in_interval: bool,
    reliability: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("📊 ML Model Validation Backtest");
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
    println!("🔬 Validating ML prediction intervals...\n");

    let min_training_window = 30;
    let mut validations = Vec::new();
    let mut interval_widths = Vec::new();

    // Walk forward through the data
    for i in min_training_window..(full_history.len() - 1) {
        let training_data = &full_history[..i];
        let actual_next_price = full_history[i + 1].price;
        let current_price = full_history[i].price;

        // Make prediction using ML model
        if let Ok(ml_forecast) = prediction_sdk::analysis::predict_next_price_ml(training_data) {
            let predicted_price = ml_forecast.predicted_price;
            let reliability = ml_forecast.reliability;

            // Calculate price bounds from return bounds
            let lower_bound = current_price * ml_forecast.lower_return.exp();
            let upper_bound = current_price * ml_forecast.upper_return.exp();

            // Check if actual price falls within the interval
            let in_interval = actual_next_price >= lower_bound && actual_next_price <= upper_bound;

            let interval_width = upper_bound - lower_bound;
            interval_widths.push(interval_width);

            validations.push(IntervalValidation {
                timestamp: full_history[i + 1].timestamp,
                actual_price: actual_next_price,
                predicted_price,
                lower_bound,
                upper_bound,
                in_interval,
                reliability,
            });
        }
    }

    if validations.is_empty() {
        eprintln!("❌ No successful validations");
        return Ok(());
    }

    // Calculate overall metrics
    let total_predictions = validations.len();
    let in_interval_count = validations.iter().filter(|v| v.in_interval).count();
    let coverage_rate = (in_interval_count as f64 / total_predictions as f64) * 100.0;

    let avg_interval_width = interval_widths.iter().sum::<f64>() / interval_widths.len() as f64;
    let avg_reliability = validations.iter().map(|v| v.reliability as f64).sum::<f64>()
        / validations.len() as f64;

    // Calculate coverage by reliability buckets
    let high_rel = validations.iter().filter(|v| v.reliability >= 0.7).collect::<Vec<_>>();
    let med_rel = validations.iter().filter(|v| v.reliability >= 0.4 && v.reliability < 0.7).collect::<Vec<_>>();
    let low_rel = validations.iter().filter(|v| v.reliability < 0.4).collect::<Vec<_>>();

    println!("📈 Interval Validation Results");
    println!("==============================");
    println!("Total Predictions:       {}", total_predictions);
    println!("In-Interval Count:       {}", in_interval_count);
    println!("Coverage Rate:           {:.1}% (target: ~90%)", coverage_rate);
    println!("Average Interval Width:  ${:.2}", avg_interval_width);
    println!("Average ML Reliability:  {:.1}%", avg_reliability * 100.0);

    // Analyze coverage by confidence levels
    println!("\n📊 Coverage by Reliability Level");
    println!("=================================");

    if !high_rel.is_empty() {
        let high_coverage = (high_rel.iter().filter(|v| v.in_interval).count() as f64 / high_rel.len() as f64) * 100.0;
        let high_avg_width = high_rel.iter().map(|v| v.upper_bound - v.lower_bound).sum::<f64>() / high_rel.len() as f64;
        println!("High (≥70%):   {} predictions, coverage: {:.1}%, avg width: ${:.2}", 
            high_rel.len(), high_coverage, high_avg_width);
    }

    if !med_rel.is_empty() {
        let med_coverage = (med_rel.iter().filter(|v| v.in_interval).count() as f64 / med_rel.len() as f64) * 100.0;
        let med_avg_width = med_rel.iter().map(|v| v.upper_bound - v.lower_bound).sum::<f64>() / med_rel.len() as f64;
        println!("Medium (40-70%): {} predictions, coverage: {:.1}%, avg width: ${:.2}", 
            med_rel.len(), med_coverage, med_avg_width);
    }

    if !low_rel.is_empty() {
        let low_coverage = (low_rel.iter().filter(|v| v.in_interval).count() as f64 / low_rel.len() as f64) * 100.0;
        let low_avg_width = low_rel.iter().map(|v| v.upper_bound - v.lower_bound).sum::<f64>() / low_rel.len() as f64;
        println!("Low (<40%):    {} predictions, coverage: {:.1}%, avg width: ${:.2}", 
            low_rel.len(), low_coverage, low_avg_width);
    }

    // Show calibration - does high reliability correlate with high accuracy?
    println!("\n🎯 Calibration Analysis");
    println!("=======================");
    
    // Calculate prediction error by reliability bucket
    let high_errors: Vec<f64> = high_rel.iter()
        .map(|v| (v.predicted_price - v.actual_price).abs())
        .collect();
    let med_errors: Vec<f64> = med_rel.iter()
        .map(|v| (v.predicted_price - v.actual_price).abs())
        .collect();
    let low_errors: Vec<f64> = low_rel.iter()
        .map(|v| (v.predicted_price - v.actual_price).abs())
        .collect();

    if !high_errors.is_empty() {
        let high_mae = high_errors.iter().sum::<f64>() / high_errors.len() as f64;
        println!("High Reliability MAE:   ${:.2}", high_mae);
    }

    if !med_errors.is_empty() {
        let med_mae = med_errors.iter().sum::<f64>() / med_errors.len() as f64;
        println!("Medium Reliability MAE: ${:.2}", med_mae);
    }

    if !low_errors.is_empty() {
        let low_mae = low_errors.iter().sum::<f64>() / low_errors.len() as f64;
        println!("Low Reliability MAE:    ${:.2}", low_mae);
    }

    // Check if reliability is well-calibrated (high reliability should have lower MAE)
    println!("\n📋 Sample Interval Validations (Last 10)");
    println!("==========================================");
    println!("{:<20} {:>9} {:>10} {:>10} {:>10} {:>6} {:>6}",
        "Timestamp", "Actual", "Predicted", "Lower", "Upper", "In?", "Rel%");
    println!("{}", "=".repeat(85));

    let sample_start = validations.len().saturating_sub(10);
    for val in &validations[sample_start..] {
        let in_symbol = if val.in_interval { "✓" } else { "✗" };
        println!(
            "{:<20} ${:>8.2} ${:>9.2} ${:>9.2} ${:>9.2} {:>6} {:>5.0}",
            val.timestamp.format("%Y-%m-%d %H:%M"),
            val.actual_price,
            val.predicted_price,
            val.lower_bound,
            val.upper_bound,
            in_symbol,
            val.reliability * 100.0
        );
    }

    // Final assessment
    println!("\n✅ Validation Assessment");
    println!("========================");
    
    if coverage_rate >= 85.0 && coverage_rate <= 95.0 {
        println!("✅ GOOD: Coverage rate ({:.1}%) is close to target (90%)", coverage_rate);
    } else if coverage_rate < 85.0 {
        println!("⚠️  WARNING: Coverage rate ({:.1}%) is below target - intervals may be too narrow", coverage_rate);
    } else {
        println!("⚠️  WARNING: Coverage rate ({:.1}%) is above target - intervals may be too wide", coverage_rate);
    }

    // Check calibration
    if !high_errors.is_empty() && !low_errors.is_empty() {
        let high_mae = high_errors.iter().sum::<f64>() / high_errors.len() as f64;
        let low_mae = low_errors.iter().sum::<f64>() / low_errors.len() as f64;
        
        if high_mae < low_mae {
            println!("✅ GOOD: Model is well-calibrated - higher reliability correlates with lower error");
        } else {
            println!("⚠️  WARNING: Model calibration needs improvement - reliability doesn't correlate with accuracy");
        }
    }

    println!("\n✅ Validation complete!");
    Ok(())
}
