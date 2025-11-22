use std::sync::{OnceLock, RwLock};

use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::Kernels;
use smartcore::svm::svr::{SVR, SVRParameters};
use ta::Next;
use ta::indicators::{BollingerBands, ExponentialMovingAverage, RelativeStrengthIndex};

use crate::analysis_deep;
use crate::dto::{
    CovariatePoint, MlForecast, MlModelConfig, MlModelKind, PredictionError, PricePoint,
    TechnicalSignals,
};
use crate::helpers;

pub(crate) const ROLLING_STANDARDIZATION_WINDOW: usize = 20;
const TARGET_COVERAGE: f64 = 0.9;

static ML_MODEL_CONFIG: OnceLock<RwLock<MlModelConfig>> = OnceLock::new();

pub(crate) fn rolling_stats(values: &[f64], end_idx: usize, window: usize) -> (f64, f64) {
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

pub(crate) fn standardize_features(raw_features: &[Vec<f64>], window: usize) -> Vec<Vec<f64>> {
    if raw_features.is_empty() {
        return Vec::new();
    }

    let feature_count = raw_features
        .first()
        .map(|row| row.len())
        .unwrap_or_default();

    if raw_features
        .iter()
        .any(|row| row.len() != feature_count || feature_count == 0)
    {
        return Vec::new();
    }

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

pub(crate) fn standardize_row(raw_row: &[f64], columns: &[Vec<f64>], window: usize) -> Vec<f64> {
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

pub fn set_ml_model_config(config: MlModelConfig) {
    let guard = ML_MODEL_CONFIG.get_or_init(|| RwLock::new(MlModelConfig::default()));
    if let Ok(mut writer) = guard.write() {
        *writer = config;
    }
}

fn active_ml_model_config() -> MlModelConfig {
    let guard = ML_MODEL_CONFIG.get_or_init(|| RwLock::new(MlModelConfig::default()));
    guard
        .read()
        .map(|config| config.clone())
        .unwrap_or_default()
}

fn rolling_out_of_fold_residuals(
    standardized_features: &[Vec<f64>],
    targets: &[f64],
    params: &SVRParameters<f64>,
) -> Result<Vec<f64>, PredictionError> {
    let mut residuals = Vec::new();
    let min_training = 10.min(standardized_features.len().saturating_sub(1));

    if standardized_features.len() < 2 || targets.len() < 2 {
        return Ok(residuals);
    }

    for idx in min_training..standardized_features.len() {
        let train_x = DenseMatrix::from_2d_vec(&standardized_features[..idx].to_vec());
        let train_y = targets[..idx].to_vec();
        if train_x.shape().0 == 0 || train_y.is_empty() {
            continue;
        }

        let model = SVR::fit(&train_x, &train_y, params)
            .map_err(|e| PredictionError::Serialization(format!("ML training failed: {}", e)))?;
        let validation_matrix = DenseMatrix::from_2d_vec(&vec![standardized_features[idx].clone()]);
        let predictions = model
            .predict(&validation_matrix)
            .map_err(|e| PredictionError::Serialization(format!("ML validation failed: {}", e)))?;
        let prediction = predictions.first().copied().unwrap_or(0.0);
        let residual = (prediction - targets[idx]).abs();
        residuals.push(residual);
    }

    Ok(residuals)
}

pub fn predict_next_price_ml(history: &[PricePoint]) -> Result<MlForecast, PredictionError> {
    predict_next_price_ml_with_covariates(history, None)
}

/// Predict the next price step using the configured ML backend.
///
/// When `covariates` are provided, they are aligned by timestamp and passed to
/// the lightweight MixLinear encoder for contextual signals; otherwise the
/// baseline price/volume features are used. The active model and hyperparameters
/// are selected via [`set_ml_model_config`].
pub fn predict_next_price_ml_with_covariates(
    history: &[PricePoint],
    covariates: Option<&[CovariatePoint]>,
) -> Result<MlForecast, PredictionError> {
    let config = active_ml_model_config();

    match config.model {
        MlModelKind::MixLinear => {
            analysis_deep::predict_next_price_ml_with_covariates(history, covariates, &config)
        }
        MlModelKind::LinearSvr => predict_next_price_svr(history),
    }
}

fn predict_next_price_svr(history: &[PricePoint]) -> Result<MlForecast, PredictionError> {
    if history.len() < 30 {
        return Err(PredictionError::InsufficientData);
    }

    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    let mut price_returns = Vec::new();

    // We need a window to compute indicators before we can start training
    let lookback = 14;
    let mut rsi =
        RelativeStrengthIndex::new(lookback).map_err(|_| PredictionError::InsufficientData)?;
    let mut bb = BollingerBands::new(20, 2.0).map_err(|_| PredictionError::InsufficientData)?;
    let mut ema_short =
        ExponentialMovingAverage::new(12).map_err(|_| PredictionError::InsufficientData)?;
    let mut ema_long =
        ExponentialMovingAverage::new(26).map_err(|_| PredictionError::InsufficientData)?;

    let mut previous_macd = 0.0;

    // Pre-warm indicators
    for point in history.iter().take(lookback) {
        rsi.next(point.price);
        let _ = bb.next(point.price);
        let _ = ema_short.next(point.price);
        let _ = ema_long.next(point.price);
    }

    for i in lookback..history.len() - 1 {
        let prev_price = history[i].price;
        let prev_prev_price = history[i - 1].price;
        let current_rsi = rsi.next(prev_price);
        let bb_out = bb.next(prev_price);
        let bb_width = (bb_out.upper - bb_out.lower) / bb_out.average;
        let short = ema_short.next(prev_price);
        let long = ema_long.next(prev_price);
        let macd = short - long;
        let macd_slope = macd - previous_macd;
        previous_macd = macd;

        let log_return_1 = (prev_price / prev_prev_price).ln();
        let log_return_2 = (history[i - 1].price / history[i - 2].price).ln();

        price_returns.push(log_return_1);
        let volatility_window_start = price_returns
            .len()
            .saturating_sub(ROLLING_STANDARDIZATION_WINDOW);
        let volatility_slice = &price_returns[volatility_window_start..];
        let (_, vol_std_dev) = rolling_stats(
            volatility_slice,
            volatility_slice.len().saturating_sub(1),
            ROLLING_STANDARDIZATION_WINDOW,
        );
        let volatility = if vol_std_dev.abs() < f64::EPSILON {
            1e-6
        } else {
            vol_std_dev
        };

        let raw_volume = history[i].volume.unwrap_or(0.0).max(0.0);
        let log_volume = (raw_volume + 1.0).ln();
        let volume_volatility_ratio = raw_volume / volatility;

        let trend_strength = (macd.abs() / prev_price) * 100.0;

        x_train.push(vec![
            log_return_1,
            log_return_2,
            current_rsi / 100.0,
            log_volume,
            volume_volatility_ratio,
            bb_width,
            macd_slope,
            trend_strength,
        ]);

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

    let params = SVRParameters::default()
        .with_c(10.0)
        .with_eps(0.1)
        .with_kernel(Kernels::linear());
    let mut residuals = rolling_out_of_fold_residuals(&standardized_features, &y_train, &params)?;

    let mae = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().copied().sum::<f64>() / residuals.len() as f64
    };

    if residuals.is_empty() {
        residuals.push(mae.abs());
    }

    let conformal_quantile = helpers::percentile(residuals.clone(), TARGET_COVERAGE)?;
    let observed_coverage = helpers::coverage_rate(&residuals, conformal_quantile);
    let pinball_loss = helpers::pinball_loss(&residuals, TARGET_COVERAGE, conformal_quantile);
    let calibration_score =
        helpers::calibration_score(observed_coverage, TARGET_COVERAGE, pinball_loss);

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
    let last_bb_out = bb.next(last_price);
    let last_bb_width = (last_bb_out.upper - last_bb_out.lower) / last_bb_out.average;
    let last_short = ema_short.next(last_price);
    let last_long = ema_long.next(last_price);
    let last_macd = last_short - last_long;
    let last_macd_slope = last_macd - previous_macd;
    let last_trend_strength = (last_macd.abs() / last_price) * 100.0;

    price_returns.push(last_log_ret);
    let volatility_window_start = price_returns
        .len()
        .saturating_sub(ROLLING_STANDARDIZATION_WINDOW);
    let volatility_slice = &price_returns[volatility_window_start..];
    let (_, vol_std_dev) = rolling_stats(
        volatility_slice,
        volatility_slice.len().saturating_sub(1),
        ROLLING_STANDARDIZATION_WINDOW,
    );
    let volatility = if vol_std_dev.abs() < f64::EPSILON {
        1e-6
    } else {
        vol_std_dev
    };

    let last_volume = history[last_idx].volume.unwrap_or(0.0).max(0.0);
    let last_log_volume = (last_volume + 1.0).ln();
    let last_volume_ratio = last_volume / volatility;

    let columns: Vec<Vec<f64>> = (0..x_train[0].len())
        .map(|col_idx| x_train.iter().map(|row| row[col_idx]).collect())
        .collect();
    let standardized_row = standardize_row(
        &[
            last_log_ret,
            prev_last_log_ret,
            last_rsi,
            last_log_volume,
            last_volume_ratio,
            last_bb_width,
            last_macd_slope,
            last_trend_strength,
        ],
        &columns,
        ROLLING_STANDARDIZATION_WINDOW,
    );

    let x_new = DenseMatrix::from_2d_vec(&vec![standardized_row]);
    let predicted_log_return = full_model
        .predict(&x_new)
        .map_err(|e| PredictionError::Serialization(format!("ML prediction failed: {}", e)))?[0];

    let lower_return = predicted_log_return - conformal_quantile;
    let upper_return = predicted_log_return + conformal_quantile;
    let interval_width = (upper_return - lower_return).abs();

    // Convert back to price
    let predicted_price = last_price * predicted_log_return.exp();

    Ok(MlForecast {
        predicted_price,
        predicted_return: predicted_log_return,
        lower_return,
        upper_return,
        calibration_score,
        target_coverage: TARGET_COVERAGE,
        observed_coverage,
        interval_width,
        pinball_loss,
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
