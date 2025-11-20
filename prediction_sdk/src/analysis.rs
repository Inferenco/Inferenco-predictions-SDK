use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::Kernels;
use smartcore::svm::svr::{SVR, SVRParameters};
use ta::Next;
use ta::indicators::{BollingerBands, ExponentialMovingAverage, RelativeStrengthIndex};

use crate::dto::{MlForecast, PredictionError, PricePoint, TechnicalSignals};

const ROLLING_STANDARDIZATION_WINDOW: usize = 20;

fn rolling_stats(values: &[f64], end_idx: usize, window: usize) -> (f64, f64) {
    let window_start = end_idx.saturating_add(1).saturating_sub(window);
    let slice = &values[window_start..=end_idx];
    let mean = if slice.is_empty() {
        0.0
    } else {
        slice.iter().sum::<f64>() / slice.len() as f64
    };
    let variance = if slice.len() < 2 {
        0.0
    } else {
        slice
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / slice.len() as f64
    };
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

fn standardize_features(raw_features: &[Vec<f64>], window: usize) -> Vec<Vec<f64>> {
    if raw_features.is_empty() {
        return Vec::new();
    }

    let feature_count = raw_features
        .first()
        .map(|row| row.len())
        .unwrap_or_default();

    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(raw_features.len()); feature_count];
    for row in raw_features {
        for (col_idx, value) in row.iter().enumerate() {
            if let Some(column) = columns.get_mut(col_idx) {
                column.push(*value);
            }
        }
    }

    let mut standardized = Vec::with_capacity(raw_features.len());
    for row_idx in 0..raw_features.len() {
        let mut transformed_row = Vec::with_capacity(feature_count);
        for column in columns.iter() {
            let (mean, std_dev) = rolling_stats(column, row_idx, window);
            let baseline = if std_dev.abs() < f64::EPSILON {
                1e-6
            } else {
                std_dev
            };
            transformed_row.push((column[row_idx] - mean) / baseline);
        }
        standardized.push(transformed_row);
    }

    standardized
}

fn standardize_row(raw_row: &[f64], columns: &[Vec<f64>], window: usize) -> Vec<f64> {
    let mut transformed_row = Vec::with_capacity(raw_row.len());
    for (col_idx, value) in raw_row.iter().enumerate() {
        let column = columns.get(col_idx).cloned().unwrap_or_default();
        let last_idx = column.len().saturating_sub(1);
        let (mean, std_dev) = if column.is_empty() {
            (0.0, 1.0)
        } else {
            rolling_stats(&column, last_idx, window)
        };
        let baseline = if std_dev.abs() < f64::EPSILON {
            1e-6
        } else {
            std_dev
        };
        transformed_row.push((*value - mean) / baseline);
    }
    transformed_row
}

/// Train a lightweight SVR model on the provided history to predict the next price step.
/// Returns the predicted next price.
pub fn predict_next_price_ml(history: &[PricePoint]) -> Result<MlForecast, PredictionError> {
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

        x_train.push(vec![log_return_1, log_return_2, current_rsi / 100.0]);

        let target_return = (history[i + 1].price / prev_price).ln();
        y_train.push(target_return);
    }

    if x_train.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    let standardized_features = standardize_features(&x_train, ROLLING_STANDARDIZATION_WINDOW);

    if standardized_features.len() < 6 {
        return Err(PredictionError::InsufficientData);
    }

    let mut split_idx = (standardized_features.len() as f64 * 0.8).ceil() as usize;
    if split_idx >= standardized_features.len() {
        split_idx = standardized_features.len() - 1;
    }

    let train_x = &standardized_features[..split_idx];
    let train_y = &y_train[..split_idx];

    let validation_x = &standardized_features[split_idx..];
    let validation_y = &y_train[split_idx..];

    if train_x.is_empty() || validation_x.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    let train_matrix = DenseMatrix::from_2d_vec(&train_x.to_vec());
    let train_targets = train_y.to_vec();
    let params = SVRParameters::default()
        .with_c(10.0)
        .with_eps(0.1)
        .with_kernel(Kernels::linear());
    let trained_model = SVR::fit(&train_matrix, &train_targets, &params)
        .map_err(|e| PredictionError::Serialization(format!("ML training failed: {}", e)))?;

    let validation_matrix = DenseMatrix::from_2d_vec(&validation_x.to_vec());
    let validation_predictions = trained_model.predict(&validation_matrix).map_err(|e| {
        PredictionError::Serialization(format!("ML validation failed: {}", e))
    })?;

    let mut mae = 0.0;
    for (predicted, actual) in validation_predictions
        .iter()
        .zip(validation_y.iter())
    {
        mae += (predicted - actual).abs();
    }
    mae /= validation_y.len() as f64;

    if !mae.is_finite() {
        return Err(PredictionError::InsufficientData);
    }

    let reliability = (1.0 / (1.0 + mae * 50.0)).clamp(0.0, 1.0) as f32;

    let full_matrix = DenseMatrix::from_2d_vec(&standardized_features);
    let full_model = SVR::fit(&full_matrix, &y_train, &params)
        .map_err(|e| PredictionError::Serialization(format!("ML training failed: {}", e)))?;

    // Train SVR
    // Predict next step
    let last_idx = history.len() - 1;
    let last_price = history[last_idx].price;
    let prev_last_price = history[last_idx - 1].price;
    let last_log_ret = (last_price / prev_last_price).ln();
    let prev_last_log_ret = (prev_last_price / history[last_idx - 2].price).ln();
    let last_rsi = rsi.next(last_price) / 100.0;

    let columns: Vec<Vec<f64>> = (0..x_train[0].len())
        .map(|col_idx| x_train.iter().map(|row| row[col_idx]).collect())
        .collect();
    let standardized_row = standardize_row(
        &[last_log_ret, prev_last_log_ret, last_rsi],
        &columns,
        ROLLING_STANDARDIZATION_WINDOW,
    );

    let x_new = DenseMatrix::from_2d_vec(&vec![standardized_row]);
    let predicted_log_return = full_model
        .predict(&x_new)
        .map_err(|e| PredictionError::Serialization(format!("ML prediction failed: {}", e)))?[0];

    // Convert back to price
    let predicted_price = last_price * predicted_log_return.exp();

    Ok(MlForecast {
        predicted_price,
        reliability,
    })
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
