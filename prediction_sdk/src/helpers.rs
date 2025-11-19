use std::cmp::Ordering;

use rand::Rng;
use rand_distr::StandardNormal;
use statrs::statistics::Statistics;

use crate::dto::{
    ForecastDecomposition,
    LongForecastHorizon,
    PredictionError,
    PricePoint,
    SentimentSnapshot,
    ShortForecastHorizon,
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

pub(crate) fn calculate_volatility(prices: &[PricePoint]) -> Result<f64, PredictionError> {
    if prices.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }

    let mut returns = Vec::with_capacity(prices.len() - 1);
    for pair in prices.windows(2) {
        let previous = pair[0].price;
        let current = pair[1].price;
        if previous <= 0.0 || current <= 0.0 {
            return Err(PredictionError::InsufficientData);
        }
        let ret = (current / previous).ln();
        returns.push(ret);
    }

    if returns.is_empty() {
        return Err(PredictionError::InsufficientData);
    }

    Ok(returns.std_dev())
}

pub(crate) fn run_monte_carlo(
    prices: &[PricePoint],
    days: u32,
    simulations: usize,
) -> Result<Vec<f64>, PredictionError> {
    let last_price = prices.last().map(|p| p.price).ok_or(PredictionError::InsufficientData)?;
    let volatility = calculate_volatility(prices)?;
    if volatility <= f64::EPSILON {
        return Ok(vec![last_price; simulations]);
    }

    let drift = returns_mean(prices)?;
    let mut rng = rand::thread_rng();
    let mut outcomes = Vec::with_capacity(simulations);
    for _ in 0..simulations {
        let mut price = last_price;
        for _ in 0..days {
            let z: f64 = rng.sample(StandardNormal);
            let step = drift - (volatility.powi(2) / 2.0) + volatility * z;
            price *= step.exp();
        }
        outcomes.push(price);
    }

    Ok(outcomes)
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

fn returns_mean(prices: &[PricePoint]) -> Result<f64, PredictionError> {
    if prices.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for pair in prices.windows(2) {
        let previous = pair[0].price;
        let current = pair[1].price;
        if previous <= 0.0 || current <= 0.0 {
            return Err(PredictionError::InsufficientData);
        }
        returns.push((current / previous).ln());
    }
    Ok(returns.mean())
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

pub(crate) fn decompose_series(prices: &[PricePoint]) -> Result<ForecastDecomposition, PredictionError> {
    if prices.len() < 2 {
        return Err(PredictionError::InsufficientData);
    }

    let trend = prices.iter().map(|p| p.price).sum::<f64>() / prices.len() as f64;
    let momentum = prices.last().map(|p| p.price).unwrap_or_default()
        - prices.first().map(|p| p.price).unwrap_or_default();
    let noise = calculate_volatility(prices)?;

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

        let result = run_monte_carlo(&history, 3, 5).unwrap();

        assert_eq!(result, vec![100.0; 5]);
    }

    #[test]
    fn run_monte_carlo_produces_expected_count() {
        let history = sample_prices(10, 100.0, 1.0);

        let result = run_monte_carlo(&history, 2, 8).unwrap();

        assert_eq!(result.len(), 8);
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
        assert_eq!(short_horizon_window(ShortForecastHorizon::FifteenMinutes), 16);
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
}
