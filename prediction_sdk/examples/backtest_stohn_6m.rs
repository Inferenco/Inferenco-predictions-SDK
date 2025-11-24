use chrono::{Duration, Utc};
use prediction_sdk::{LongForecastHorizon, PredictionSdk};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Stohn Coin 6-Month Backtest...");

    let sdk = PredictionSdk::new()?;
    let asset_id = "stohn-coin";
    let vs_currency = "usd";

    // 1. Fetch 1 year of data (365 days)
    // We need enough history to have a "training" period and a "testing" period.
    // For a 6-month forecast, we want at least 6 months of history prior to the forecast start.
    // So let's fetch 1 year: First 6 months = History, Next 6 months = Future (Test)
    println!("📊 Fetching 1 year of historical data for {}...", asset_id);
    let now = Utc::now();
    let one_year_ago = now - Duration::days(365);

    let full_history = sdk
        .fetch_price_history_range(asset_id, vs_currency, one_year_ago, now)
        .await?;

    if full_history.is_empty() {
        eprintln!("❌ Failed to fetch history.");
        return Ok(());
    }

    // 2. Split Data
    // We want to forecast from 6 months ago.
    // Split point is roughly the middle of the dataset.
    let split_idx = full_history.len() / 2;
    let train_data = &full_history[..split_idx];
    let test_data = &full_history[split_idx..];

    let forecast_start_date = train_data.last().unwrap().timestamp;
    let test_end_date = test_data.last().unwrap().timestamp;

    println!("📅 Data Split:");
    println!(
        "   Training: {} points (Ends: {})",
        train_data.len(),
        forecast_start_date
    );
    println!(
        "   Testing:  {} points (Ends: {})",
        test_data.len(),
        test_end_date
    );

    // 3. Run Forecast
    println!("🔮 Running 6-Month Forecast...");
    let (forecast, projection) = sdk
        .run_long_forecast(
            train_data,
            LongForecastHorizon::SixMonths,
            None,
            true, // We need the chart projection to compare against daily prices
        )
        .await?;

    let projection = projection.ok_or("Forecast did not return a projection")?;

    // 4. Analyze Results
    println!("📉 Analyzing Performance...");

    let mut total_error = 0.0;
    let mut points_checked = 0;
    let mut points_in_bounds = 0;

    println!("\n📅 Daily Accuracy Report:");
    println!(
        "{:<12} | {:<10} | {:<10} | {:<10} | {:<8}",
        "Date", "Actual", "Predicted", "Error", "In Bounds"
    );
    println!(
        "{:-<12}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<8}",
        "", "", "", "", ""
    );

    // Map projection by date for easy lookup (ignoring time for daily granularity)
    // The projection steps are daily.
    for point in test_data {
        // Find the corresponding projection point
        // We look for a projection point with the same date
        if let Some(proj) = projection
            .iter()
            .find(|p| p.timestamp.date_naive() == point.timestamp.date_naive())
        {
            let error = (proj.mean - point.price).abs();
            total_error += error;

            let in_bounds = point.price >= proj.percentile_10 && point.price <= proj.percentile_90;
            if in_bounds {
                points_in_bounds += 1;
            }

            points_checked += 1;

            // Print daily step
            println!(
                "{:<12} | ${:<9.4} | ${:<9.4} | ${:<9.4} | {}",
                point.timestamp.format("%Y-%m-%d"),
                point.price,
                proj.mean,
                error,
                if in_bounds { "✅" } else { "❌" }
            );
        }
    }

    if points_checked == 0 {
        println!("⚠️ No overlapping dates found between forecast and test data.");
        return Ok(());
    }

    let mae = total_error / points_checked as f64;
    let coverage = (points_in_bounds as f64 / points_checked as f64) * 100.0;

    // Directional Accuracy
    let start_price = train_data.last().unwrap().price;
    let end_price_actual = test_data.last().unwrap().price;
    let end_price_predicted = forecast.mean_price;

    let actual_direction = if end_price_actual > start_price {
        "UP"
    } else {
        "DOWN"
    };
    let predicted_direction = if end_price_predicted > start_price {
        "UP"
    } else {
        "DOWN"
    };
    let direction_correct = actual_direction == predicted_direction;

    println!("\n📋 Backtest Results (Stohn Coin 6-Month Horizon)");
    println!("==========================================");
    println!("Start Price:       ${:.4}", start_price);
    println!("End Price (Act):   ${:.4}", end_price_actual);
    println!("End Price (Pred):  ${:.4}", end_price_predicted);
    println!("------------------------------------------");
    println!("MAE:               ${:.4}", mae);
    println!("Coverage (P10-P90): {:.1}%", coverage);
    println!(
        "Direction:         Predicted {}, Actual {} ({})",
        predicted_direction,
        actual_direction,
        if direction_correct {
            "✅ Correct"
        } else {
            "❌ Incorrect"
        }
    );
    println!("==========================================");

    Ok(())
}
