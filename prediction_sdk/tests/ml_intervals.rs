use chrono::{Duration, Utc};
use prediction_sdk::{PredictionSdk, PricePoint, ShortForecastHorizon, analysis};

fn exponential_history(count: usize, step: f64) -> Vec<PricePoint> {
    let start = Utc::now() - Duration::minutes(count as i64);
    (0..count)
        .map(|idx| {
            let timestamp = start + Duration::minutes(idx as i64);
            let price = 100.0 * (step * idx as f64).exp();
            PricePoint {
                timestamp,
                price,
                volume: Some(1_000.0 + idx as f64),
            }
        })
        .collect()
}

fn volatile_history(count: usize) -> Vec<PricePoint> {
    let start = Utc::now() - Duration::minutes(count as i64);
    (0..count)
        .map(|idx| {
            let timestamp = start + Duration::minutes(idx as i64);
            let phase = (idx as f64 / 5.0).sin();
            let jump = if idx % 12 == 0 { 0.15 } else { -0.05 * phase };
            let baseline = 200.0 + (idx % 10) as f64;
            let price = (baseline * (1.0 + jump)).max(1.0);

            PricePoint {
                timestamp,
                price,
                volume: Some(500.0 + (idx % 7) as f64 * 10.0),
            }
        })
        .collect()
}

fn moving_average(prices: &[PricePoint], window: usize) -> f64 {
    let start = prices.len().saturating_sub(window);
    let slice = &prices[start..];
    let sum: f64 = slice.iter().map(|point| point.price).sum();
    sum / window as f64
}

fn drift_and_volatility(prices: &[PricePoint]) -> (f64, f64) {
    if prices.len() < 2 {
        return (0.0, 0.0);
    }

    let mut returns = Vec::with_capacity(prices.len() - 1);
    for pair in prices.windows(2) {
        if let [prev, current] = pair {
            let delta = match (current.timestamp - prev.timestamp).to_std() {
                Ok(delta) => delta,
                Err(_) => return (0.0, 0.0),
            };
            let delta_days = delta.as_secs_f64() / 86_400.0;
            if delta_days <= f64::EPSILON || prev.price <= 0.0 || current.price <= 0.0 {
                return (0.0, 0.0);
            }
            returns.push((current.price / prev.price).ln() / delta_days);
        }
    }

    if returns.is_empty() {
        return (0.0, 0.0);
    }

    let mean = returns.iter().copied().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    (mean, variance.sqrt())
}

#[tokio::test]
async fn conformal_intervals_cover_synthetic_returns() {
    let history = exponential_history(140, 0.001);
    let mut covered = 0usize;
    let mut total = 0usize;

    for idx in 50..history.len() - 1 {
        let forecast =
            analysis::predict_next_price_ml(&history[..idx]).expect("ML forecast should succeed");
        let actual_return = (history[idx].price / history[idx - 1].price).ln();
        if actual_return >= forecast.lower_return && actual_return <= forecast.upper_return {
            covered += 1;
        }
        total += 1;
    }

    let coverage = covered as f64 / total as f64;
    assert!(coverage >= 0.8, "coverage {} below target", coverage);
}

#[tokio::test]
async fn wide_intervals_downweight_ml_blending() {
    let history = volatile_history(120);
    let sdk = PredictionSdk::new().expect("SDK should construct");

    let result = sdk
        .run_short_forecast(&history, ShortForecastHorizon::OneHour, None)
        .await
        .expect("short forecast should succeed");

    let ml_prediction = result
        .ml_prediction
        .expect("ml prediction should be present");
    let ml_reliability = result
        .ml_reliability
        .expect("ml reliability should be present");
    let price_interval = result
        .ml_price_interval
        .expect("ml price interval should exist");

    let (drift, _volatility) = drift_and_volatility(&history);
    let time_fraction = 1.0 / 24.0;
    let trend_adjustment = (drift * time_fraction).exp();
    let base = moving_average(&history, 48) * trend_adjustment;

    let weight_inferred = if (ml_prediction - base).abs() < f64::EPSILON {
        0.0
    } else {
        (result.expected_price - base) / (ml_prediction - base)
    };

    let interval_width = (price_interval.1 - price_interval.0).abs();
    let relative_width = interval_width / price_interval.0.max(price_interval.1).abs();

    assert!(weight_inferred >= 0.0);
    assert!(
        relative_width > 0.05,
        "interval too narrow to test weighting"
    );
    assert!(weight_inferred < f64::from(ml_reliability));
}
