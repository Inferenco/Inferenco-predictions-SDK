use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use itertools::Itertools;
use ta::Next;
use ta::indicators::{BollingerBands, ExponentialMovingAverage, RelativeStrengthIndex};

use crate::analysis::{rolling_stats, standardize_features, ROLLING_STANDARDIZATION_WINDOW};
use crate::dto::{CovariatePoint, MlForecast, MlModelConfig, PredictionError, PricePoint};
use crate::helpers;

const TARGET_COVERAGE: f64 = 0.9;

#[derive(Clone, Debug)]
struct MixLinearModel {
    weights: Vec<Vec<f64>>, // component -> weight vector
    biases: Vec<f64>,
}

fn covariate_length(point: &CovariatePoint) -> usize {
    point.macro_covariates.len() + point.onchain_covariates.len() + point.sentiment_covariates.len()
}

fn covariate_map(covariates: Option<&[CovariatePoint]>) -> (BTreeMap<DateTime<Utc>, CovariatePoint>, usize) {
    let mut map = BTreeMap::new();
    let mut max_len = 0;

    if let Some(points) = covariates {
        for point in points {
            max_len = max_len.max(covariate_length(point));
            map.insert(point.timestamp, point.clone());
        }
    }

    (map, max_len)
}

fn covariate_vector(
    timestamp: &DateTime<Utc>,
    map: &BTreeMap<DateTime<Utc>, CovariatePoint>,
    expected_len: usize,
) -> Vec<f64> {
    let mut values = Vec::new();
    if let Some(point) = map.get(timestamp) {
        values.extend_from_slice(&point.macro_covariates);
        values.extend_from_slice(&point.onchain_covariates);
        values.extend_from_slice(&point.sentiment_covariates);
    }

    if expected_len > values.len() {
        values.extend(std::iter::repeat(0.0).take(expected_len - values.len()));
    }

    values
}

fn build_feature_rows(
    history: &[PricePoint],
    covariates: Option<&[CovariatePoint]>,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), PredictionError> {
    if history.len() < 30 {
        return Err(PredictionError::InsufficientData);
    }

    let (cov_map, cov_len) = covariate_map(covariates);

    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    let mut price_returns = Vec::new();

    let lookback = 14;
    let mut rsi = RelativeStrengthIndex::new(lookback).map_err(|_| PredictionError::InsufficientData)?;
    let mut bb = BollingerBands::new(20, 2.0).map_err(|_| PredictionError::InsufficientData)?;
    let mut ema_short = ExponentialMovingAverage::new(12).map_err(|_| PredictionError::InsufficientData)?;
    let mut ema_long = ExponentialMovingAverage::new(26).map_err(|_| PredictionError::InsufficientData)?;

    let mut previous_macd = 0.0;

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

        let mut features = vec![
            log_return_1,
            log_return_2,
            current_rsi / 100.0,
            log_volume,
            volume_volatility_ratio,
            bb_width,
            macd_slope,
            trend_strength,
        ];

        if cov_len > 0 {
            features.extend(covariate_vector(&history[i].timestamp, &cov_map, cov_len));
        }

        x_train.push(features);

        let target_return = (history[i + 1].price / prev_price).ln();
        y_train.push(target_return);
    }

    if x_train.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    Ok((x_train, y_train))
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();
    let sum: f64 = exp_values.iter().sum();
    if sum.abs() < f64::EPSILON {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exp_values.iter().map(|value| value / sum).collect()
}

fn predict_mix_linear(model: &MixLinearModel, sample: &[f64]) -> f64 {
    if model.weights.is_empty() {
        return 0.0;
    }

    let mut logits = Vec::with_capacity(model.weights.len());
    let sample_mean = sample.iter().copied().sum::<f64>() / sample.len().max(1) as f64;
    for (idx, weights) in model.weights.iter().enumerate() {
        let dot = weights
            .iter()
            .zip(sample.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>()
            + model.biases[idx];
        let gate = dot + (idx as f64 + 1.0) * sample_mean;
        logits.push(gate);
    }

    let mixture = softmax(&logits);
    model
        .weights
        .iter()
        .zip(model.biases.iter())
        .zip(mixture.iter())
        .map(|((weights, bias), weight)| {
            let projection = weights
                .iter()
                .zip(sample.iter())
                .map(|(w, s)| w * s)
                .sum::<f64>()
                + bias;
            projection * weight
        })
        .sum()
}

fn train_mix_linear(
    inputs: &[Vec<f64>],
    targets: &[f64],
    config: &MlModelConfig,
) -> MixLinearModel {
    let input_dim = inputs
        .first()
        .map(|row| row.len())
        .unwrap_or_default();
    let components = config.mixture_components.max(1);
    let learning_rate = (config.learning_rate).clamp(1e-4, 0.05);

    let mut weights = vec![vec![0.0; input_dim]; components];
    let mut biases = vec![0.0; components];

    for epoch in 0..30 {
        for (sample, target) in inputs.iter().zip(targets.iter()) {
            let prediction = predict_mix_linear(
                &MixLinearModel {
                    weights: weights.clone(),
                    biases: biases.clone(),
                },
                sample,
            );
            let error = prediction - target;
            let lr = learning_rate / (1.0 + epoch as f64 * 0.25);
            for (component_idx, (component_weights, bias)) in
                weights.iter_mut().zip(biases.iter_mut()).enumerate()
            {
                let gate = 1.0 / (component_idx as f64 + 1.0);
                for (w, s) in component_weights.iter_mut().zip(sample.iter()) {
                    *w -= lr * (error * s * gate + 0.0005 * *w);
                }
                *bias -= lr * error * gate;
            }
        }
    }

    MixLinearModel { weights, biases }
}

fn rolling_validation(
    inputs: &[Vec<f64>],
    targets: &[f64],
    config: &MlModelConfig,
) -> (MixLinearModel, Vec<f64>, f64) {
    let mut residuals = Vec::new();
    let mut mae_sum = 0.0;
    let mut mae_count = 0;

    let window = config.validation_window.max(2);
    let stride = config.validation_stride.max(1);

    if inputs.len() <= window {
        let model = train_mix_linear(inputs, targets, config);
        return (model, residuals, 0.0);
    }

    for start in (window..inputs.len()).step_by(stride) {
        let validation_end = (start + window).min(inputs.len());
        let train_inputs = &inputs[..start];
        let train_targets = &targets[..start];
        let validation_inputs = &inputs[start..validation_end];
        let validation_targets = &targets[start..validation_end];

        if train_inputs.is_empty() || validation_inputs.is_empty() {
            continue;
        }

        let model = train_mix_linear(train_inputs, train_targets, config);
        for (sample, target) in validation_inputs.iter().zip(validation_targets.iter()) {
            let pred = predict_mix_linear(&model, sample);
            let residual = (pred - target).abs();
            residuals.push(residual);
            mae_sum += residual;
            mae_count += 1;
        }
    }

    let mut final_model = train_mix_linear(inputs, targets, config);

    if mae_count > 0 {
        let mae = mae_sum / mae_count as f64;
        (final_model, residuals, mae)
    } else {
        final_model = train_mix_linear(inputs, targets, config);
        (final_model, residuals, 0.0)
    }
}

fn build_patches(
    standardized_rows: &[Vec<f64>],
    targets: &[f64],
    patch_length: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut patches = Vec::new();
    let mut patch_targets = Vec::new();

    if patch_length == 0 {
        return (patches, patch_targets);
    }

    let offset = patch_length.saturating_sub(1);
    for idx in offset..standardized_rows.len().min(targets.len()) {
        if idx + 1 > standardized_rows.len() || idx >= targets.len() {
            break;
        }
        if idx + 1 < patch_length {
            continue;
        }
        let start = idx + 1 - patch_length;
        let window = &standardized_rows[start..=idx];
        let flattened: Vec<f64> = window.iter().flatten().copied().collect();
        patches.push(flattened);
        patch_targets.push(targets[idx]);
    }

    (patches, patch_targets)
}

pub(crate) fn predict_next_price_ml_with_covariates(
    history: &[PricePoint],
    covariates: Option<&[CovariatePoint]>,
    config: &MlModelConfig,
) -> Result<MlForecast, PredictionError> {
    let (feature_rows, targets) = build_feature_rows(history, covariates)?;
    let standardized_rows = standardize_features(&feature_rows, ROLLING_STANDARDIZATION_WINDOW);

    if standardized_rows.len() <= config.patch_length || targets.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    let (patches, patch_targets) =
        build_patches(&standardized_rows, &targets, config.patch_length.max(2));

    if patches.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    let (model, residuals, mae) = rolling_validation(&patches, &patch_targets, config);

    let mut forecast_window: Vec<Vec<f64>> = standardized_rows
        .iter()
        .rev()
        .take(config.patch_length)
        .cloned()
        .collect_vec();
    forecast_window.reverse();
    if forecast_window.len() < config.patch_length {
        return Err(PredictionError::InsufficientData);
    }
    let flattened: Vec<f64> = forecast_window.into_iter().flatten().collect();
    let predicted_return = predict_mix_linear(&model, &flattened);

    let mut residuals_clone = residuals.clone();
    if residuals_clone.is_empty() {
        residuals_clone.push(mae.abs());
    }

    let conformal_quantile = helpers::percentile(residuals_clone.clone(), TARGET_COVERAGE)?;
    let observed_coverage = helpers::coverage_rate(&residuals_clone, conformal_quantile);
    let pinball_loss = helpers::pinball_loss(&residuals_clone, TARGET_COVERAGE, conformal_quantile);
    let calibration_score =
        helpers::calibration_score(observed_coverage, TARGET_COVERAGE, pinball_loss);

    let lower_return = predicted_return - conformal_quantile;
    let upper_return = predicted_return + conformal_quantile;
    let interval_width = (upper_return - lower_return).abs();

    let last_price = history
        .last()
        .map(|point| point.price)
        .ok_or(PredictionError::InsufficientData)?;
    let predicted_price = last_price * predicted_return.exp();

    Ok(MlForecast {
        predicted_price,
        predicted_return,
        lower_return,
        upper_return,
        calibration_score,
        target_coverage: TARGET_COVERAGE,
        observed_coverage,
        interval_width,
        pinball_loss,
    })
}
