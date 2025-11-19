use std::collections::HashMap;

use chrono::{DateTime, Duration, TimeZone, Utc};
use reqwest::{Client, StatusCode};
use serde::Deserialize;

use crate::helpers;
use crate::{
    ForecastHorizon,
    ForecastResult,
    LongForecastHorizon,
    LongForecastResult,
    PredictionError,
    PricePoint,
    SentimentSnapshot,
    ShortForecastHorizon,
    ShortForecastResult,
};

const DEFAULT_BASE_URL: &str = "https://api.coingecko.com/api/v3";
const DEFAULT_SIMULATIONS: usize = 256;
const SHORT_FORECAST_LOOKBACK_DAYS: u32 = 30;

pub struct PredictionSdk {
    client: Client,
    market_base_url: String,
}

impl PredictionSdk {
    pub fn new() -> Result<Self, PredictionError> {
        let client = Client::builder()
            .build()
            .map_err(|err| PredictionError::Network(err.to_string()))?;
        Ok(Self {
            client,
            market_base_url: DEFAULT_BASE_URL.to_string(),
        })
    }

    pub fn with_client(client: Client, market_base_url: Option<String>) -> Self {
        let url = market_base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Self {
            client,
            market_base_url: url,
        }
    }

    pub async fn fetch_price_history(
        &self,
        asset_id: &str,
        vs_currency: &str,
        days: u32,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        let url = format!("{}/coins/{}/market_chart", self.market_base_url, asset_id);
        let query = vec![
            ("vs_currency", vs_currency.to_string()),
            ("days", days.to_string()),
        ];

        self.request_market_chart(url, query).await
    }

    pub async fn fetch_market_chart_range(
        &self,
        id: &str,
        vs_currency: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        let url = format!(
            "{}/coins/{}/market_chart/range",
            self.market_base_url, id
        );
        let query = vec![
            ("vs_currency", vs_currency.to_string()),
            ("from", from.timestamp().to_string()),
            ("to", to.timestamp().to_string()),
        ];

        self.request_market_chart(url, query).await
    }

    pub async fn run_short_forecast(
        &self,
        history: &[PricePoint],
        horizon: ShortForecastHorizon,
        sentiment: Option<SentimentSnapshot>,
    ) -> Result<ShortForecastResult, PredictionError> {
        let window = helpers::short_horizon_window(horizon);
        let moving_average = helpers::calculate_moving_average(history, window)?;
        let volatility = helpers::calculate_volatility(history)?;
        let mut expected_price = moving_average + volatility * 0.1;
        if let Some(snapshot) = sentiment.as_ref() {
            expected_price = helpers::weight_with_sentiment(expected_price, snapshot);
        }

        let decomposition = helpers::decompose_series(history)?;
        let confidence_value = (volatility + decomposition.noise).recip().abs();
        let confidence = helpers::normalize_confidence(confidence_value);

        Ok(ShortForecastResult {
            horizon,
            expected_price,
            confidence,
            decomposition,
        })
    }

    pub async fn run_long_forecast(
        &self,
        history: &[PricePoint],
        horizon: LongForecastHorizon,
        sentiment: Option<SentimentSnapshot>,
    ) -> Result<LongForecastResult, PredictionError> {
        let days = helpers::long_horizon_days(horizon);
        let simulations = helpers::scaled_simulation_count(days, DEFAULT_SIMULATIONS);
        let paths = helpers::run_monte_carlo(history, days, simulations)?;
        let mean_price = paths.iter().sum::<f64>() / paths.len() as f64;
        let percentile_10 = helpers::percentile(paths.clone(), 0.1)?;
        let percentile_90 = helpers::percentile(paths, 0.9)?;
        let mut adjusted_mean = mean_price;
        if let Some(snapshot) = sentiment.as_ref() {
            adjusted_mean = helpers::weight_with_sentiment(adjusted_mean, snapshot);
        }
        let confidence = helpers::normalize_confidence(1.0 / (1.0 + days as f64));

        Ok(LongForecastResult {
            horizon,
            mean_price: adjusted_mean,
            percentile_10,
            percentile_90,
            confidence,
        })
    }

    pub async fn forecast(
        &self,
        history: &[PricePoint],
        horizon: ForecastHorizon,
        sentiment: Option<SentimentSnapshot>,
    ) -> Result<ForecastResult, PredictionError> {
        match horizon {
            ForecastHorizon::Short(short_horizon) => self
                .run_short_forecast(history, short_horizon, sentiment)
                .await
                .map(ForecastResult::Short),
            ForecastHorizon::Long(long_horizon) => self
                .run_long_forecast(history, long_horizon, sentiment)
                .await
                .map(ForecastResult::Long),
        }
    }

    pub async fn forecast_with_fetch(
        &self,
        asset_id: &str,
        vs_currency: &str,
        horizon: ForecastHorizon,
        sentiment: Option<SentimentSnapshot>,
    ) -> Result<ForecastResult, PredictionError> {
        match horizon {
            ForecastHorizon::Short(short_horizon) => {
                let history = self
                    .fetch_price_history(asset_id, vs_currency, SHORT_FORECAST_LOOKBACK_DAYS)
                    .await?;
                self
                    .forecast(
                        &history,
                        ForecastHorizon::Short(short_horizon),
                        sentiment,
                    )
                    .await
            }
            ForecastHorizon::Long(long_horizon) => {
                let now = Utc::now();
                let lookback_days = helpers::long_horizon_days(long_horizon);
                let start = now - Duration::days(i64::from(lookback_days));
                let history = self
                    .fetch_market_chart_range(asset_id, vs_currency, start, now)
                    .await?;
                self
                    .forecast(
                        &history,
                        ForecastHorizon::Long(long_horizon),
                        sentiment,
                    )
                    .await
            }
        }
    }

    async fn request_market_chart(
        &self,
        url: String,
        query: Vec<(&str, String)>,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        let response = self
            .client
            .get(url)
            .query(&query)
            .send()
            .await
            .map_err(|err| PredictionError::Network(err.to_string()))?;

        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            return Err(PredictionError::Network(
                "rate limited by upstream provider".to_string(),
            ));
        }

        if !response.status().is_success() {
            return Err(PredictionError::Network(format!(
                "unexpected status: {}",
                response.status()
            )));
        }

        let payload: MarketChartResponse = response
            .json()
            .await
            .map_err(|err| PredictionError::Serialization(err.to_string()))?;

        build_price_points(payload)
    }
}

#[derive(Deserialize)]
struct MarketChartResponse {
    prices: Vec<[f64; 2]>,
    #[serde(default)]
    total_volumes: Vec<[f64; 2]>,
}

fn build_price_points(payload: MarketChartResponse) -> Result<Vec<PricePoint>, PredictionError> {
    let mut volumes_map = payload
        .total_volumes
        .iter()
        .map(|entry| (entry[0] as i64, entry[1]))
        .collect::<HashMap<_, _>>();

    let mut points = Vec::with_capacity(payload.prices.len());
    for entry in payload.prices {
        let timestamp_ms = entry[0] as i64;
        let price = entry[1];
        let timestamp = Utc
            .timestamp_millis_opt(timestamp_ms)
            .single()
            .ok_or(PredictionError::TimeConversion)?;
        let volume = volumes_map.remove(&timestamp_ms);
        points.push(PricePoint {
            timestamp,
            price,
            volume,
        });
    }

    Ok(points)
}
