use std::cmp::Ordering;

use rand::Rng;
use rand_distr::StandardNormal;
use statrs::statistics::Statistics;

use crate::dto::{
    ForecastDecomposition, LongForecastHorizon, PredictionError, PricePoint, SentimentSnapshot,
    MonteCarloBenchmark, ShortForecastHorizon,
};

pub(crate) fn short_horizon_window(horizon: ShortForecastHorizon) -> usize {
    match horizon {
        ShortForecastHorizon::FifteenMinutes => 16,
        ShortForecastHorizon::OneHour => 48,
        ShortForecastHorizon::FourHours => 96,
    }
}

pub(crate) fn long_horizon_days(horizon: LongForecastHorizon) -> u32 {
    match horizon {
        LongForecastHorizon::OneDay => 1,
        LongForecastHorizon::ThreeDays => 3,
        LongForecastHorizon::OneWeek => 7,
        LongForecastHorizon::OneMonth => 30,
        LongForecastHorizon::ThreeMonths => 3 * 30,
        LongForecastHorizon::SixMonths => 6 * 30,
        LongForecastHorizon::OneYear => 12 * 30,
        LongForecastHorizon::FourYears => 48 * 30,
    }
}

pub(crate) fn scaled_simulation_count(days: u32, base: usize) -> usize {
    let normalized_days = days.max(30);
    let scaling = (30.0f64 / normalized_days as f64).sqrt();
    let scaled = (base as f64 * scaling).round() as usize;
    scaled.clamp(32, base)
}

fn ordered_history(prices: &[PricePoint]) -> Result<Vec<&PricePoint>, PredictionError> {
    if prices.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }

    let mut ordered: Vec<&PricePoint> = prices.iter().collect();
    ordered.sort_by_key(|point| point.timestamp);

    Ok(ordered)
}

fn latest_price(prices: &[PricePoint]) -> Result<f64, PredictionError> {
    let ordered = ordered_history(prices)?;

    ordered
        .last()
        .map(|point| point.price)
        .ok_or(PredictionError::InsufficientData)
}

fn daily_log_returns(prices: &[PricePoint]) -> Result<Vec<f64>, PredictionError> {
    let ordered = ordered_history(prices)?;
    let mut returns = Vec::with_capacity(ordered.len() - 1);

    for pair in ordered.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if previous.price <= 0.0 || current.price <= 0.0 {
            return Err(PredictionError::InsufficientData);
        }

        let delta = (current.timestamp - previous.timestamp)
            .to_std()
            .map_err(|_| PredictionError::InsufficientData)?;
        let delta_days = delta.as_secs_f64() / 86_400.0;
        if delta_days <= f64::EPSILON {
            return Err(PredictionError::InsufficientData);
        }

        let ret = (current.price / previous.price).ln() / delta_days;
        returns.push(ret);
    }

    if returns.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    Ok(returns)
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn garch_parameters(returns: &[f64]) -> (f64, f64, f64) {
    let alpha = 0.12_f64;
    let beta = 0.83_f64;
    let base_variance = variance(returns);
    let floor = 1e-6;
    let omega = (1.0_f64 - alpha - beta).max(floor) * base_variance.max(floor);

    (omega, alpha, beta)
}

pub(crate) fn forecast_volatility_series(
    prices: &[PricePoint],
    steps: u32,
) -> Result<Vec<f64>, PredictionError> {
    let returns = daily_log_returns(prices)?;
    if returns.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }

    let (omega, alpha, beta) = garch_parameters(&returns);
    let mut sigma_squared = variance(&returns).max(1e-6);
    let mut residual_squared = returns.last().map(|last| last.powi(2)).unwrap_or(0.0);

    let mut path = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        sigma_squared = omega + alpha * residual_squared + beta * sigma_squared;
        let sigma = sigma_squared.sqrt();
        path.push(sigma);
        residual_squared = 0.0;
    }

    Ok(path)
}

pub(crate) fn daily_return_stats(
    prices: &[PricePoint],
) -> Result<(f64, f64), PredictionError> {
    let returns = daily_log_returns(prices)?;
    let volatility = if returns.len() < 2 {
        0.0
    } else {
        returns.clone().std_dev()
    };
    let drift = returns.mean();

    Ok((drift, volatility))
}

pub(crate) fn calculate_moving_average(
    prices: &[PricePoint],
    window: usize,
) -> Result<f64, PredictionError> {
    if prices.len() < window || window == 0 {
        return Err(PredictionError::InsufficientData);
    }

    let start = prices.len() - window;
    let slice = &prices[start..];
    let sum: f64 = slice.iter().map(|p| p.price).sum();
    Ok(sum / window as f64)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn calculate_volatility(prices: &[PricePoint]) -> Result<f64, PredictionError> {
    if prices.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }

    let returns = daily_log_returns(prices)?;

    if returns.len() < 2 {
        return Ok(0.0);
    }

    Ok(returns.std_dev())
}

pub(crate) fn run_monte_carlo(
    prices: &[PricePoint],
    days: u32,
    simulations: usize,
    drift: f64,
    volatility_path: &[f64],
    mean_reversion: Option<f64>,
) -> Result<Vec<f64>, PredictionError> {
    if volatility_path.len() < days as usize {
        return Err(PredictionError::InsufficientData);
    }

    let last_price = latest_price(prices)?;
    let mut rng = rand::thread_rng();
    let mut outcomes = Vec::with_capacity(simulations);
    let time_step = 1.0;
    for _ in 0..simulations {
        let mut price = last_price;
        for day in 0..days as usize {
            let volatility = *volatility_path
                .get(day)
                .ok_or(PredictionError::InsufficientData)?;
            let z: f64 = rng.sample(StandardNormal);
            let drift_component = (drift - (volatility.powi(2) / 2.0)) * time_step;
            let reversion_adjustment = mean_reversion
                .map(|kappa| kappa * ((last_price - price) / last_price))
                .unwrap_or(0.0);
            let shock = volatility * time_step.sqrt() * z;
            let step = drift_component + reversion_adjustment + shock;
            price *= step.exp();
        }
        outcomes.push(price);
    }

    Ok(outcomes)
}

pub(crate) fn monte_carlo_benchmark(
    prices: &[PricePoint],
    days: u32,
    simulations: usize,
    drift: f64,
    mean_reversion: Option<f64>,
) -> Result<MonteCarloBenchmark, PredictionError> {
    let constant_volatility = calculate_volatility(prices)?;
    let constant_path = vec![constant_volatility; days as usize];
    let constant_runs = run_monte_carlo(
        prices,
        days,
        simulations,
        drift,
        &constant_path,
        mean_reversion,
    )?;

    let dynamic_volatility = forecast_volatility_series(prices, days)?;
    let regime_runs = run_monte_carlo(
        prices,
        days,
        simulations,
        drift,
        &dynamic_volatility,
        mean_reversion,
    )?;

    let constant_mean = constant_runs.iter().sum::<f64>() / constant_runs.len() as f64;
    let constant_percentile_10 = percentile(constant_runs.clone(), 0.1)?;
    let constant_percentile_90 = percentile(constant_runs.clone(), 0.9)?;

    let regime_mean = regime_runs.iter().sum::<f64>() / regime_runs.len() as f64;
    let regime_percentile_10 = percentile(regime_runs.clone(), 0.1)?;
    let regime_percentile_90 = percentile(regime_runs.clone(), 0.9)?;

    Ok(MonteCarloBenchmark {
        horizon_days: days,
        constant_mean,
        constant_percentile_10,
        constant_percentile_90,
        regime_mean,
        regime_percentile_10,
        regime_percentile_90,
    })
}

pub(crate) fn percentile(mut values: Vec<f64>, pct: f64) -> Result<f64, PredictionError> {
    if values.is_empty() {
        return Err(PredictionError::InsufficientData);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let len = values.len();
    let idx = (((len - 1) as f64) * pct).round() as usize;
    values
        .get(idx.min(len - 1))
        .copied()
        .ok_or(PredictionError::InsufficientData)
}

pub(crate) fn weight_with_sentiment(value: f64, sentiment: &SentimentSnapshot) -> f64 {
    let bounded_news = sentiment.news_score.clamp(-1.0, 1.0);
    let bounded_social = sentiment.social_score.clamp(-1.0, 1.0);
    let adjustment = (bounded_news + bounded_social) / 10.0;
    value * (1.0 + adjustment)
}

pub(crate) fn normalize_confidence(value: f64) -> f32 {
    value.clamp(0.0, 1.0) as f32
}

pub(crate) fn decompose_series(
    prices: &[PricePoint],
) -> Result<ForecastDecomposition, PredictionError> {
    if prices.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }

    let momentum = prices.last().map(|p| p.price).unwrap_or_default()
        - prices.first().map(|p| p.price).unwrap_or_default();

    let price_values: Vec<f64> = prices.iter().map(|p| p.price).collect();
    let count = price_values.len();
    let indices: Vec<f64> = (0..count).map(|idx| idx as f64).collect();
    let mean_x = indices.iter().copied().sum::<f64>() / count as f64;
    let mean_y = price_values.iter().copied().sum::<f64>() / count as f64;

    let denominator: f64 = indices
        .iter()
        .map(|x| (x - mean_x).powi(2))
        .sum();
    let slope = if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        indices
            .iter()
            .zip(price_values.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / denominator
    };
    let intercept = mean_y - slope * mean_x;
    let trend = intercept + slope * (count as f64 - 1.0);

    let residuals: Vec<f64> = indices
        .iter()
        .zip(price_values.iter())
        .map(|(x, price)| price - (intercept + slope * x))
        .collect();
    let noise = if residuals.len() < 2 {
        0.0
    } else {
        residuals.std_dev()
    };

    Ok(ForecastDecomposition {
        trend,
        momentum,
        noise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn sample_prices(count: usize, start_price: f64, step: f64) -> Vec<PricePoint> {
        (0..count)
            .map(|idx| PricePoint {
                timestamp: Utc::now() - Duration::minutes(idx as i64),
                price: start_price + step * idx as f64,
                volume: None,
            })
            .collect()
    }

    #[test]
    fn run_monte_carlo_returns_constant_when_no_volatility() {
        let history = vec![
            PricePoint {
                timestamp: Utc::now(),
                price: 100.0,
                volume: None,
            },
            PricePoint {
                timestamp: Utc::now() + Duration::minutes(1),
                price: 100.0,
                volume: None,
            },
        ];

        let (drift, volatility) = daily_return_stats(&history).unwrap();
        let volatility_path = vec![volatility; 3];
        let result = run_monte_carlo(&history, 3, 5, drift, &volatility_path, None).unwrap();

        assert_eq!(result, vec![100.0; 5]);
    }

    #[test]
    fn run_monte_carlo_produces_expected_count() {
        let history = sample_prices(10, 100.0, 1.0);

        let (drift, volatility) = daily_return_stats(&history).unwrap();

        let volatility_path = vec![volatility; 2];

        let result = run_monte_carlo(&history, 2, 8, drift, &volatility_path, Some(0.01)).unwrap();

        assert_eq!(result.len(), 8);
    }

    #[test]
    fn time_scaled_returns_respect_elapsed_intervals() {
        let start = Utc::now() - Duration::hours(24);
        let history = vec![
            PricePoint {
                timestamp: start,
                price: 100.0,
                volume: None,
            },
            PricePoint {
                timestamp: start + Duration::hours(12),
                price: 110.0,
                volume: None,
            },
            PricePoint {
                timestamp: start + Duration::hours(24),
                price: 121.0,
                volume: None,
            },
        ];

        let (drift, volatility) = daily_return_stats(&history).unwrap();
        let expected_drift = (1.1_f64.ln()) / 0.5;

        assert!((drift - expected_drift).abs() < 1e-6);
        assert!(volatility.abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_sorts_and_selects_value() {
        let values = vec![5.0, 1.0, 3.0, 2.0, 4.0];

        let p50 = percentile(values, 0.5).unwrap();

        assert_eq!(p50, 3.0);
    }

    #[test]
    fn percentile_errors_on_empty_input() {
        let result = percentile(Vec::new(), 0.9);

        assert!(matches!(result, Err(PredictionError::InsufficientData)));
    }

    #[test]
    fn horizon_to_lookback_mapping_matches_expected() {
        assert_eq!(
            short_horizon_window(ShortForecastHorizon::FifteenMinutes),
            16
        );
        assert_eq!(short_horizon_window(ShortForecastHorizon::OneHour), 48);
        assert_eq!(short_horizon_window(ShortForecastHorizon::FourHours), 96);

        assert_eq!(long_horizon_days(LongForecastHorizon::OneDay), 1);
        assert_eq!(long_horizon_days(LongForecastHorizon::ThreeDays), 3);
        assert_eq!(long_horizon_days(LongForecastHorizon::OneWeek), 7);
        assert_eq!(long_horizon_days(LongForecastHorizon::OneMonth), 30);
        assert_eq!(long_horizon_days(LongForecastHorizon::ThreeMonths), 90);
        assert_eq!(long_horizon_days(LongForecastHorizon::SixMonths), 180);
        assert_eq!(long_horizon_days(LongForecastHorizon::OneYear), 360);
        assert_eq!(long_horizon_days(LongForecastHorizon::FourYears), 1440);
    }
    #[test]
    fn calculate_moving_average_errors_on_insufficient_data() {
        let history = sample_prices(5, 100.0, 1.0);
        let result = calculate_moving_average(&history, 10);
        assert!(matches!(result, Err(PredictionError::InsufficientData)));
    }

    #[test]
    fn calculate_volatility_handles_flat_prices() {
        let history = sample_prices(10, 100.0, 0.0);
        let volatility = calculate_volatility(&history).unwrap();
        assert!(volatility.abs() < f64::EPSILON);
    }

    #[test]
    fn calculate_volatility_errors_on_insufficient_data() {
        let history = sample_prices(1, 100.0, 0.0);
        let result = calculate_volatility(&history);
        assert!(matches!(result, Err(PredictionError::InsufficientData)));
    }

    #[test]
    fn run_monte_carlo_errors_on_insufficient_data() {
        let history = sample_prices(1, 100.0, 0.0);
        let volatility_path = vec![0.0; 5];
        let result = run_monte_carlo(&history, 5, 10, 0.0, &volatility_path, None);
        assert!(matches!(result, Err(PredictionError::InsufficientData)));
    }

    #[test]
    fn forecast_volatility_series_uses_variance_floor() {
        let history = sample_prices(5, 100.0, 0.0);
        let result = forecast_volatility_series(&history, 4).unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|sigma| *sigma >= 0.0));
    }

    #[test]
    fn monte_carlo_benchmark_reports_bands() {
        let history = sample_prices(20, 100.0, 1.0);
        let (drift, _) = daily_return_stats(&history).unwrap();
        let benchmark =
            monte_carlo_benchmark(&history, 5, 12, drift, Some(0.02)).unwrap();

        assert!(benchmark.constant_mean.is_finite());
        assert!(benchmark.regime_mean.is_finite());
        assert!(benchmark.constant_percentile_10 <= benchmark.constant_percentile_90);
        assert!(benchmark.regime_percentile_10 <= benchmark.regime_percentile_90);
    }
}
