use chrono::{Duration, Utc};
use prediction_sdk::PredictionSdk;
use std::error::Error;

#[derive(Debug, Clone)]
struct Trade {
    timestamp: chrono::DateTime<chrono::Utc>,
    action: TradeAction,
    price: f64,
    reason: String,
}

#[derive(Debug, Clone, PartialEq)]
enum TradeAction {
    Buy,
    Sell,
}

#[derive(Debug)]
struct Position {
    has_position: bool,
    entry_price: f64,
    entry_timestamp: chrono::DateTime<chrono::Utc>,
}

impl Position {
    fn new() -> Self {
        Position {
            has_position: false,
            entry_price: 0.0,
            entry_timestamp: Utc::now(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("💹 Trading Strategy Backtest");
    println!("============================\n");

    let sdk = PredictionSdk::new()?;
    let asset = "bitcoin";
    let currency = "usd";
    let lookback_days = 30;

    // Strategy parameters
    let price_threshold = 0.005; // 0.5% price difference to trigger trade
    let min_confidence = 0.5; // Minimum ML reliability to consider trade

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
    println!("🎯 Strategy Parameters:");
    println!("   Price Threshold:     {:.1}%", price_threshold * 100.0);
    println!("   Min ML Confidence:   {:.0}%\n", min_confidence * 100.0);
    println!("🔬 Running trading simulation...\n");

    let min_training_window = 30;
    let mut position = Position::new();
    let mut trades = Vec::new();
    let mut portfolio_value = Vec::new();
    let initial_cash = 10000.0;
    let mut cash = initial_cash;
    let mut shares = 0.0;

    // Track for statistics
    let mut winning_trades = 0;
    let mut losing_trades = 0;

    // Walk forward through the data
    for i in min_training_window..(full_history.len() - 1) {
        let training_data = &full_history[..i];
        let current_price = full_history[i].price;
        let timestamp = full_history[i].timestamp;

        // Calculate current portfolio value
        let current_portfolio_value = cash + (shares * current_price);
        portfolio_value.push((timestamp, current_portfolio_value));

        // Make prediction using ML model
        if let Ok(ml_forecast) = prediction_sdk::analysis::predict_next_price_ml(training_data) {
            let predicted_price = ml_forecast.predicted_price;
            let reliability = ml_forecast.reliability;

            // Skip if confidence is too low
            if reliability < min_confidence {
                continue;
            }

            let price_diff_pct = (predicted_price - current_price) / current_price;

            // Trading logic
            if !position.has_position && price_diff_pct > price_threshold {
                // BUY signal: predicted price is significantly higher
                shares = cash / current_price;
                cash = 0.0;
                position.has_position = true;
                position.entry_price = current_price;
                position.entry_timestamp = timestamp;

                trades.push(Trade {
                    timestamp,
                    action: TradeAction::Buy,
                    price: current_price,
                    reason: format!(
                        "ML predicts {:.1}% gain (conf: {:.0}%)",
                        price_diff_pct * 100.0,
                        reliability * 100.0
                    ),
                });
            } else if position.has_position && price_diff_pct < -price_threshold {
                // SELL signal: predicted price is significantly lower
                cash = shares * current_price;
                let profit = cash - (position.entry_price * shares);
                if profit > 0.0 {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }
                shares = 0.0;
                position.has_position = false;

                trades.push(Trade {
                    timestamp,
                    action: TradeAction::Sell,
                    price: current_price,
                    reason: format!(
                        "ML predicts {:.1}% drop (conf: {:.0}%), P/L: ${:.2}",
                        price_diff_pct * 100.0,
                        reliability * 100.0,
                        profit
                    ),
                });
            }
        }
    }

    // Close any open position at the end
    if position.has_position {
        let final_price = full_history.last().unwrap().price;
        cash = shares * final_price;
        let profit = cash - (position.entry_price * shares);
        if profit > 0.0 {
            winning_trades += 1;
        } else {
            losing_trades += 1;
        }
        shares = 0.0;

        trades.push(Trade {
            timestamp: full_history.last().unwrap().timestamp,
            action: TradeAction::Sell,
            price: final_price,
            reason: format!("Position closed at end of backtest, P/L: ${:.2}", profit),
        });
    }

    // Calculate final metrics
    let final_value = cash + (shares * full_history.last().unwrap().price);
    let total_return = ((final_value - initial_cash) / initial_cash) * 100.0;
    let total_trades = trades.len() / 2; // Each round trip is 2 trades
    let win_rate = if total_trades > 0 {
        (winning_trades as f64 / (winning_trades + losing_trades) as f64) * 100.0
    } else {
        0.0
    };

    // Calculate buy-and-hold baseline
    let initial_price = full_history[min_training_window].price;
    let final_price = full_history.last().unwrap().price;
    let buy_hold_return = ((final_price - initial_price) / initial_price) * 100.0;

    // Calculate max drawdown
    let mut max_value = initial_cash;
    let mut max_drawdown = 0.0;
    for (_, value) in &portfolio_value {
        if *value > max_value {
            max_value = *value;
        }
        let drawdown = ((max_value - value) / max_value) * 100.0;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Print results
    println!("📈 Trading Performance");
    println!("=====================");
    println!("Initial Capital:     ${:.2}", initial_cash);
    println!("Final Value:         ${:.2}", final_value);
    println!("Total Return:        {:.2}%", total_return);
    println!("Buy & Hold Return:   {:.2}%", buy_hold_return);
    println!("Alpha:               {:.2}%", total_return - buy_hold_return);
    println!("\nTrading Statistics:");
    println!("Total Trades:        {}", total_trades);
    println!("Winning Trades:      {}", winning_trades);
    println!("Losing Trades:       {}", losing_trades);
    println!("Win Rate:            {:.1}%", win_rate);
    println!("Max Drawdown:        {:.2}%", max_drawdown);

    // Show trade log
    println!("\n📋 Trade Log (Last 10 trades)");
    println!("=====================================");
    let sample_start = trades.len().saturating_sub(10);
    for trade in &trades[sample_start..] {
        let action_symbol = match trade.action {
            TradeAction::Buy => "🟢 BUY ",
            TradeAction::Sell => "🔴 SELL",
        };
        println!(
            "{} {} at ${:.2} - {}",
            action_symbol,
            trade.timestamp.format("%Y-%m-%d %H:%M"),
            trade.price,
            trade.reason
        );
    }

    // Performance analysis
    println!("\n📊 Performance Analysis");
    println!("======================");
    if total_return > buy_hold_return {
        println!("✅ Strategy OUTPERFORMED buy-and-hold by {:.2}%", total_return - buy_hold_return);
    } else {
        println!("❌ Strategy UNDERPERFORMED buy-and-hold by {:.2}%", buy_hold_return - total_return);
    }

    if win_rate >= 50.0 {
        println!("✅ Positive win rate: {:.1}%", win_rate);
    } else {
        println!("⚠️  Low win rate: {:.1}%", win_rate);
    }

    println!("\n✅ Backtest complete!");
    Ok(())
}
