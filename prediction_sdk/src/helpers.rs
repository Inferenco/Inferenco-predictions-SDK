use std::cmp::Ordering;

use rand::distributions::{Distribution, StandardNormal};
use rand::Rng;
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
    }
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
            price *= (1.0 + step).max(0.01);
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
        returns.push((current / previous) - 1.0);
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
