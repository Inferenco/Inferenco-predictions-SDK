use chrono::{Duration, Utc};
use prediction_sdk::{PredictionSdk, PricePoint, ShortForecastHorizon};

fn build_spiky_history(count: usize) -> Vec<PricePoint> {
    let start = Utc::now() - Duration::minutes(count as i64);
    (0..count)
        .map(|idx| {
            let timestamp = start + Duration::minutes(idx as i64);
            let price = if idx % 5 == 0 {
                1_000.0 + idx as f64
            } else {
                100.0 + (idx % 3) as f64
            };

            PricePoint {
                timestamp,
                price,
                volume: None,
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
            if delta_days <= f64::EPSILON {
                return (0.0, 0.0);
            }
            if prev.price <= 0.0 || current.price <= 0.0 {
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
async fn unreliable_ml_is_downweighted() {
    let history = build_spiky_history(80);
    let sdk = PredictionSdk::new().expect("SDK should construct");

    let result = sdk
        .run_short_forecast(&history, ShortForecastHorizon::OneHour, None)
        .await
        .expect("short forecast should succeed");

    let (drift, _volatility) = drift_and_volatility(&history);
    let time_fraction = 1.0 / 24.0;
    let trend_adjustment = (drift * time_fraction).exp();
    let base = moving_average(&history, 48) * trend_adjustment;

    let calibration = result
        .ml_interval_calibration
        .expect("ml calibration should exist");
    assert!(calibration.calibration_score < 0.6);

    let diff = (result.expected_price - base).abs();
    assert!(diff < base * 0.05 + 1e-6);
}
