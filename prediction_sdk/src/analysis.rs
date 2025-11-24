use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use ta::Next;
use ta::indicators::{BollingerBands, ExponentialMovingAverage, RelativeStrengthIndex};

use crate::analysis_deep;
use crate::dto::{
    CovariatePoint, MlForecast, MlModelConfig, MlModelKind, PredictionError, PricePoint,
    TechnicalSignals,
};

pub(crate) const ROLLING_STANDARDIZATION_WINDOW: usize = 20;
const MODEL_CACHE_TTL: Duration = Duration::from_secs(60 * 60); // 1 hour

static ML_MODEL_CONFIG: OnceLock<RwLock<MlModelConfig>> = OnceLock::new();
static MODEL_CACHE: OnceLock<RwLock<HashMap<ModelCacheKey, ModelCacheEntry>>> = OnceLock::new();

#[derive(Clone, Hash, Eq, PartialEq)]
pub(crate) struct ModelCacheKey {
    kind: MlModelKind,
    config_hash: u64,
    data_hash: u64,
}

impl ModelCacheKey {
    pub(crate) fn new(kind: MlModelKind, config_hash: u64, data_hash: u64) -> Self {
        Self {
            kind,
            config_hash,
            data_hash,
        }
    }
}

#[derive(Clone)]
pub(crate) enum CachedModelData {
    Mix(MixLinearModelCache),
}

#[derive(Clone)]
pub(crate) struct MixLinearModelCache {
    pub(crate) model: analysis_deep::MixLinearModel,
    pub(crate) conformal_quantile: f64,
    pub(crate) observed_coverage: f64,
    pub(crate) pinball_loss: f64,
    pub(crate) calibration_score: f32,
    pub(crate) target_coverage: f64,
}

struct ModelCacheEntry {
    stored_at: Instant,
    model: CachedModelData,
}

fn model_cache() -> &'static RwLock<HashMap<ModelCacheKey, ModelCacheEntry>> {
    MODEL_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(crate) fn ml_config_hash(config: &MlModelConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.model.hash(&mut hasher);
    config.patch_length.hash(&mut hasher);
    config.mixture_components.hash(&mut hasher);
    config.learning_rate.to_bits().hash(&mut hasher);
    config.validation_window.hash(&mut hasher);
    config.validation_stride.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn ml_data_hash(
    history: &[PricePoint],
    covariates: Option<&[CovariatePoint]>,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    history.len().hash(&mut hasher);
    for point in history.iter().rev().take(128) {
        point.timestamp.timestamp().hash(&mut hasher);
        point.price.to_bits().hash(&mut hasher);
        point.volume.unwrap_or(0.0).to_bits().hash(&mut hasher);
    }

    if let Some(cov) = covariates {
        cov.len().hash(&mut hasher);
        for point in cov.iter().rev().take(64) {
            point.timestamp.timestamp().hash(&mut hasher);
            for value in point
                .macro_covariates
                .iter()
                .chain(point.onchain_covariates.iter())
                .chain(point.sentiment_covariates.iter())
            {
                value.to_bits().hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

pub(crate) fn get_cached_model(key: &ModelCacheKey) -> Option<CachedModelData> {
    let cache = model_cache();
    let now = Instant::now();
    if let Ok(mut guard) = cache.write() {
        if let Some(entry) = guard.get(key) {
            if now.duration_since(entry.stored_at) <= MODEL_CACHE_TTL {
                return Some(entry.model.clone());
            }
        }
        guard.remove(key);
    }
    None
}

pub(crate) fn store_cached_model(key: ModelCacheKey, model: CachedModelData) {
    if let Ok(mut guard) = model_cache().write() {
        guard.insert(
            key,
            ModelCacheEntry {
                stored_at: Instant::now(),
                model,
            },
        );
    }
}

#[cfg(test)]
pub(crate) fn clear_model_cache() {
    if let Ok(mut guard) = model_cache().write() {
        guard.clear();
    }
}

#[cfg(test)]
pub(crate) fn model_cache_len() -> usize {
    model_cache()
        .read()
        .map(|guard| guard.len())
        .unwrap_or(0)
}

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
    analysis_deep::predict_next_price_ml_with_covariates(history, covariates, &config)
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn build_history(count: usize, start_price: f64, step: f64) -> Vec<PricePoint> {
        (0..count)
            .map(|idx| PricePoint {
                timestamp: Utc::now() - Duration::minutes((count - idx) as i64),
                price: start_price + step * idx as f64,
                volume: Some(10.0 + idx as f64),
            })
            .collect()
    }

    #[test]
    fn caches_trained_mix_model() {
        clear_model_cache();
        let history = build_history(60, 100.0, 0.5);
        let config = MlModelConfig {
            model: MlModelKind::MixLinear,
            ..Default::default()
        };
        set_ml_model_config(config.clone());

        let _ = predict_next_price_ml(&history).expect("first run should succeed");
        assert!(model_cache_len() > 0, "model should be cached");

        let _ = predict_next_price_ml(&history).expect("cached run should succeed");
    }
}
