use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::Kernels;
use smartcore::svm::svr::{SVR, SVRParameters};
use ta::Next;
use ta::indicators::{BollingerBands, ExponentialMovingAverage, RelativeStrengthIndex};

use crate::dto::{PredictionError, PricePoint, TechnicalSignals};

/// Train a lightweight SVR model on the provided history to predict the next price step.
/// Returns the predicted next price.
pub fn predict_next_price_ml(history: &[PricePoint]) -> Result<f64, PredictionError> {
    if history.len() < 30 {
        return Err(PredictionError::InsufficientData);
    }

    // Feature Engineering:
    // X = [Price(t-1), Price(t-2), RSI(t-1), Volatility(t-1)]
    // y = Price(t)

    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    // We need a window to compute indicators before we can start training
    let lookback = 14;
    let mut rsi =
        RelativeStrengthIndex::new(lookback).map_err(|_| PredictionError::InsufficientData)?;

    // Pre-warm indicators
    for point in history.iter().take(lookback) {
        rsi.next(point.price);
    }

    for i in lookback..history.len() - 1 {
        let prev_price = history[i].price;
        let prev_prev_price = history[i - 1].price;
        let current_rsi = rsi.next(prev_price);

        // Simple features: Lagged prices and RSI
        // Note: In a real "best ever" system, we'd normalize these features (e.g. log returns, z-scores)
        // For this implementation, we'll use raw values but SVR might struggle with unscaled data.
        // Let's switch to log-returns for better ML stability.

        let log_return_1 = (prev_price / prev_prev_price).ln();
        let log_return_2 = (history[i - 1].price / history[i - 2].price).ln();

        x_train.push(vec![log_return_1, log_return_2, current_rsi / 100.0]); // Normalize RSI to 0-1

        let target_return = (history[i + 1].price / prev_price).ln();
        y_train.push(target_return);
    }

    if x_train.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    let x_matrix = DenseMatrix::from_2d_vec(&x_train);
    let y_vector = y_train;

    // Train SVR
    let params = SVRParameters::default()
        .with_c(10.0)
        .with_eps(0.1)
        .with_kernel(Kernels::linear());
    let svr = SVR::fit(&x_matrix, &y_vector, &params)
        .map_err(|e| PredictionError::Serialization(format!("ML training failed: {}", e)))?;

    // Predict next step
    let last_idx = history.len() - 1;
    let last_price = history[last_idx].price;
    let prev_last_price = history[last_idx - 1].price;
    let last_log_ret = (last_price / prev_last_price).ln();
    let prev_last_log_ret = (prev_last_price / history[last_idx - 2].price).ln();
    let last_rsi = rsi.next(last_price) / 100.0;

    let x_new = DenseMatrix::from_2d_vec(&vec![vec![last_log_ret, prev_last_log_ret, last_rsi]]);
    let predicted_log_return = svr
        .predict(&x_new)
        .map_err(|e| PredictionError::Serialization(format!("ML prediction failed: {}", e)))?[0];

    // Convert back to price
    let predicted_price = last_price * predicted_log_return.exp();

    Ok(predicted_price)
}

pub fn calculate_technical_signals(
    history: &[PricePoint],
) -> Result<TechnicalSignals, PredictionError> {
    if history.len() < 20 {
        return Err(PredictionError::InsufficientData);
    }

    let mut rsi = RelativeStrengthIndex::new(14).map_err(|_| PredictionError::InsufficientData)?;
    let mut bb = BollingerBands::new(20, 2.0).map_err(|_| PredictionError::InsufficientData)?;
    let mut ema_short =
        ExponentialMovingAverage::new(12).map_err(|_| PredictionError::InsufficientData)?;
    let mut ema_long =
        ExponentialMovingAverage::new(26).map_err(|_| PredictionError::InsufficientData)?;

    let mut last_rsi = 0.0;
    let mut last_bb_width = 0.0;
    let mut last_macd = 0.0;

    for point in history {
        last_rsi = rsi.next(point.price);
        let bb_out = bb.next(point.price);
        last_bb_width = (bb_out.upper - bb_out.lower) / bb_out.average;

        let short = ema_short.next(point.price);
        let long = ema_long.next(point.price);
        last_macd = short - long;
    }

    // Simple trend strength proxy: Absolute MACD normalized by price (roughly)
    let last_price = history.last().unwrap().price;
    let trend_strength = (last_macd.abs() / last_price) * 100.0;

    Ok(TechnicalSignals {
        rsi: last_rsi,
        macd_divergence: last_macd,
        bollinger_width: last_bb_width,
        trend_strength,
    })
}
