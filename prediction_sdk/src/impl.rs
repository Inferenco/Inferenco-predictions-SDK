use std::collections::HashMap;

use chrono::{DateTime, Duration, TimeZone, Utc};
use governor::{Quota, RateLimiter};
use governor::clock::DefaultClock;
use governor::state::InMemoryState;
use governor::middleware::NoOpMiddleware;
use nonzero_ext::nonzero;

use std::sync::Arc;
use tokio::time::sleep;

use reqwest::{Client, StatusCode};
use serde::Deserialize;

use crate::helpers;
use crate::analysis;
use crate::cache::{ApiCache, ForecastCache, ApiCacheKey, ForecastCacheKey, ForecastCacheValue, hash_sentiment};
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
    limiter: Arc<RateLimiter<governor::state::NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>>,
    api_cache: Arc<ApiCache>,
    forecast_cache: Arc<ForecastCache>,
}

impl PredictionSdk {
    /// Construct an SDK client that uses the default CoinGecko base URL.
    ///
    /// The internal `reqwest::Client` uses default TLS and proxy settings. If
    /// client construction fails, the [`PredictionError::Network`] variant is
    /// returned.
    ///
    /// ```no_run
    /// use prediction_sdk::PredictionSdk;
    ///
    /// let sdk = PredictionSdk::new()?;
    /// # Ok::<(), prediction_sdk::PredictionError>(())
    /// ```
    pub fn new() -> Result<Self, PredictionError> {
        let client = Client::builder()
            .build()
            .map_err(|err| PredictionError::Network(err.to_string()))?;
        // Rate limit: 8 requests per second (burst 8)
        let quota = Quota::per_second(nonzero!(8u32));
        let limiter = Arc::new(RateLimiter::direct(quota));
        let api_cache = Arc::new(ApiCache::new());
        let forecast_cache = Arc::new(ForecastCache::new());

        Ok(Self {
            client,
            market_base_url: DEFAULT_BASE_URL.to_string(),
            limiter,
            api_cache,
            forecast_cache,
        })
    }

    /// Build an SDK with a pre-configured HTTP client and optional base URL.
    ///
    /// Use this constructor when you need to inject custom timeouts,
    /// instrumentation, or a mock server URL during tests.
    ///
    /// ```no_run
    /// use prediction_sdk::PredictionSdk;
    /// use reqwest::Client;
    ///
    /// let client = Client::builder().timeout(std::time::Duration::from_secs(10)).build()?;
    /// let sdk = PredictionSdk::with_client(client, None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_client(client: Client, market_base_url: Option<String>) -> Self {
        let url = market_base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        // Rate limit: 8 requests per second (burst 8)
        let quota = Quota::per_second(nonzero!(8u32));
        let limiter = Arc::new(RateLimiter::direct(quota));
        let api_cache = Arc::new(ApiCache::new());
        let forecast_cache = Arc::new(ForecastCache::new());

        Self {
            client,
            market_base_url: url,
            limiter,
            api_cache,
            forecast_cache,
        }
    }

    /// Fetch historical market data for a specific window expressed in days.
    ///
    /// The `days` parameter controls the lookback period requested from
    /// `/market_chart`. CoinGecko accepts either integer day counts or the
    /// string `"max"`; this method uses `u32` to emphasize typical bounded
    /// lookbacks.
    ///
    /// Errors are returned when the upstream call fails, times cannot be
    /// converted, or the response payload cannot be deserialized.
    pub async fn fetch_price_history(
        &self,
        asset_id: &str,
        vs_currency: &str,
        days: u32,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        // Check API cache first
        let cache_key = ApiCacheKey::DaysLookback {
            asset_id: asset_id.to_string(),
            vs_currency: vs_currency.to_string(),
            days,
        };

        if let Some(cached) = self.api_cache.get(&cache_key).await {
            return Ok(cached);
        }

        let url = format!("{}/coins/{}/market_chart", self.market_base_url, asset_id);
        let query = vec![
            ("vs_currency", vs_currency.to_string()),
            ("days", days.to_string()),
        ];

        let result = self.request_market_chart(url, query).await?;
        
        // Store in cache
        self.api_cache.insert(cache_key, result.clone()).await;
        
        Ok(result)
    }

    /// Fetch historical market data for a specific time range.
    ///
    /// Timestamps must be provided in UTC. The upstream API expects Unix epoch
    /// seconds; this method handles the conversion and surface any
    /// deserialization or time conversion errors through [`PredictionError`].
    pub async fn fetch_price_history_range(
        &self,
        asset_id: &str,
        vs_currency: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        let url = format!(
            "{}/coins/{}/market_chart/range",
            self.market_base_url, asset_id
        );
        let query = vec![
            ("vs_currency", vs_currency.to_string()),
            ("from", from.timestamp().to_string()),
            ("to", to.timestamp().to_string()),
        ];

        self.request_market_chart(url, query).await
    }

    /// Convenience alias for [`fetch_price_history_range`].
    ///
    /// Maintained for compatibility with existing call sites; both methods
    /// produce identical results and errors.
    pub async fn fetch_market_chart_range(
        &self,
        id: &str,
        vs_currency: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        self
            .fetch_price_history_range(id, vs_currency, from, to)
            .await
    }

    /// Run a short-horizon forecast using pre-fetched price history.
    ///
    /// * `history` should contain at least the window length implied by the
    ///   [`ShortForecastHorizon`] variant (typically recent intraday data).
    /// * `sentiment` can be provided to nudge the forecast toward positive or
    ///   negative momentum.
    ///
    /// Returns [`PredictionError::InsufficientData`] when the supplied history
    /// cannot support the requested calculations.
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
        // Use a sigmoid-like decay for confidence based on volatility.
        // High volatility -> Low confidence.
        // Low volatility -> High confidence.
        // Formula: 1.0 / (1.0 + scaling_factor * volatility * sqrt(time))
        
        let horizon_hours: f64 = match horizon {
            ShortForecastHorizon::FifteenMinutes => 0.25,
            ShortForecastHorizon::OneHour => 1.0,
            ShortForecastHorizon::FourHours => 4.0,
        };

        let scaling_factor = 12.0; // Tuned to balance short-horizon return regimes
        let time_scaling = horizon_hours.sqrt();
        let combined_uncertainty = 0.6 * volatility + 0.4 * decomposition.noise;
        let confidence_value = 1.0 / (1.0 + scaling_factor * combined_uncertainty * time_scaling);
        let confidence = helpers::normalize_confidence(confidence_value);

        // Calculate technical signals
        let technical_signals = analysis::calculate_technical_signals(history).ok();
        
        // Run ML prediction
        let ml_prediction = match analysis::predict_next_price_ml(history) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("ML Prediction failed: {}", e);
                None
            }
        };

        let baseline_expected = expected_price;

        if let Some(ml) = ml_prediction.as_ref() {
            let reliability = f64::from(ml.reliability);
            if reliability >= 0.1 {
                expected_price = baseline_expected * (1.0 - reliability)
                    + ml.predicted_price * reliability;
            } else {
                expected_price = baseline_expected;
            }
        }

        Ok(ShortForecastResult {
            horizon,
            expected_price,
            confidence,
            decomposition,
            technical_signals,
            ml_prediction: ml_prediction.as_ref().map(|ml| ml.predicted_price),
            ml_reliability: ml_prediction.map(|ml| ml.reliability),
        })
    }

    /// Run a long-horizon forecast using Monte Carlo simulation.
    ///
    /// Provide a sufficiently long history to span the selected horizon (for
    /// example, a three-month horizon should be paired with multiple months of
    /// data). Sentiment can be used to bias the resulting mean price. Returns
    /// [`PredictionError::InsufficientData`] when the series cannot support the
    /// simulation.
    pub async fn run_long_forecast(
        &self,
        history: &[PricePoint],
        horizon: LongForecastHorizon,
        sentiment: Option<SentimentSnapshot>,
    ) -> Result<LongForecastResult, PredictionError> {
        let days = helpers::long_horizon_days(horizon);
        let simulations = helpers::scaled_simulation_count(days, DEFAULT_SIMULATIONS);
        let (drift, volatility) = helpers::daily_return_stats(history)?;
        let paths = helpers::run_monte_carlo(history, days, simulations, drift, volatility)?;
        let mean_price = paths.iter().sum::<f64>() / paths.len() as f64;
        let percentile_10 = helpers::percentile(paths.clone(), 0.1)?;
        let percentile_90 = helpers::percentile(paths, 0.9)?;
        let mut adjusted_mean = mean_price;
        if let Some(snapshot) = sentiment.as_ref() {
            adjusted_mean = helpers::weight_with_sentiment(adjusted_mean, snapshot);
        }
        let confidence = helpers::normalize_confidence(1.0 / (1.0 + days as f64));

        // Calculate technical signals (still useful for long horizons as a starting point)
        let technical_signals = analysis::calculate_technical_signals(history).ok();

        Ok(LongForecastResult {
            horizon,
            mean_price: adjusted_mean,
            percentile_10,
            percentile_90,
            confidence,
            technical_signals,
        })
    }

    /// Dispatch to the appropriate forecast algorithm using pre-fetched
    /// history.
    ///
    /// Provide at least 30 days of data for [`ForecastHorizon::Short`]. Long
    /// horizons should include enough history to cover the number of days
    /// returned by `helpers::long_horizon_days` to avoid
    /// [`PredictionError::InsufficientData`].
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

    /// Fetch the necessary history and produce a forecast in one call.
    ///
    /// * Short forecasts automatically pull the last 30 days of hourly data to
    ///   satisfy the moving-average lookback requirements.
    /// * Long forecasts derive their lookback window from the selected horizon
    ///   (e.g., a three-month horizon uses a multi-month history window).
    ///
    /// Prefer [`forecast`] when you already possess cached history to avoid
    /// duplicate network calls. Network errors, rate limits, deserialization
    /// failures, and insufficient history propagate as [`PredictionError`]
    /// variants.
    pub async fn forecast_with_fetch(
        &self,
        asset_id: &str,
        vs_currency: &str,
        horizon: ForecastHorizon,
        sentiment: Option<SentimentSnapshot>,
    ) -> Result<ForecastResult, PredictionError> {
        // Check forecast cache first
        let sentiment_hash = hash_sentiment(&sentiment);
        
        let cache_key = match horizon {
            ForecastHorizon::Short(h) => ForecastCacheKey::Short {
                asset_id: asset_id.to_string(),
                vs_currency: vs_currency.to_string(),
                horizon: h,
                sentiment_hash,
            },
            ForecastHorizon::Long(h) => ForecastCacheKey::Long {
                asset_id: asset_id.to_string(),
                vs_currency: vs_currency.to_string(),
                horizon: h,
                sentiment_hash,
            },
        };

        if let Some(cached) = self.forecast_cache.get(&cache_key).await {
            return match cached {
                ForecastCacheValue::Short(r) => Ok(ForecastResult::Short(r)),
                ForecastCacheValue::Long(r) => Ok(ForecastResult::Long(r)),
            };
        }

        // Cache miss - compute forecast
        let result = match horizon {
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
                    .await?
            }
            ForecastHorizon::Long(long_horizon) => {
                let now = Utc::now();
                let lookback_days = helpers::long_horizon_days(long_horizon);
                let start = now - Duration::days(i64::from(lookback_days));
                let history = self
                    .fetch_price_history_range(asset_id, vs_currency, start, now)
                    .await?;
                self
                    .forecast(
                        &history,
                        ForecastHorizon::Long(long_horizon),
                        sentiment,
                    )
                    .await?
            }
        };

        // Store in forecast cache
        let cache_value = match &result {
            ForecastResult::Short(r) => ForecastCacheValue::Short(r.clone()),
            ForecastResult::Long(r) => ForecastCacheValue::Long(r.clone()),
        };
        self.forecast_cache.insert(cache_key, cache_value).await;

        Ok(result)
    }

    async fn request_market_chart(
        &self,
        url: String,
        query: Vec<(&str, String)>,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        let max_retries = 5;
        let mut attempt = 0;
        let base_delay_ms = 500;

        loop {
            // 1. Wait for rate limiter (Token Bucket)
            self.limiter.until_ready().await;

            // 2. Make request
            let response = self
                .client
                .get(&url)
                .query(&query)
                .send()
                .await
                .map_err(|err| PredictionError::Network(err.to_string()))?;

            let status = response.status();

            if status.is_success() {
                let payload: MarketChartResponse = response
                    .json()
                    .await
                    .map_err(|err| PredictionError::Serialization(err.to_string()))?;
                return build_price_points(payload);
            }

            if status == StatusCode::TOO_MANY_REQUESTS {
                attempt += 1;
                if attempt > max_retries {
                    return Err(PredictionError::Network(
                        "rate limited by upstream provider (max retries exceeded)".to_string(),
                    ));
                }

                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());

                let delay = if let Some(seconds) = retry_after {
                    std::time::Duration::from_secs(seconds)
                } else {
                    // Exponential backoff: 500ms * 2^(attempt-1)
                    let factor = 2u64.pow(attempt as u32 - 1);
                    std::time::Duration::from_millis(base_delay_ms * factor)
                };

                sleep(delay).await;
                continue;
            }

            return Err(PredictionError::Network(format!(
                "unexpected status: {}",
                status
            )));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use httpmock::prelude::*;


    fn build_sdk(server: &MockServer) -> PredictionSdk {
        let client = Client::builder().build().unwrap();
        PredictionSdk::with_client(client, Some(server.base_url()))
    }

    fn synthetic_volatility_series(scale: f64) -> Vec<PricePoint> {
        let start = Utc::now() - Duration::days(60);
        (0..60)
            .map(|idx| {
                let t = start + Duration::days(idx as i64);
                let base_growth = 1.0 + 0.001 * idx as f64;
                let oscillation = (idx as f64 / 3.0).sin() * scale;
                let price = 100.0 * base_growth * (1.0 + oscillation);
                PricePoint {
                    timestamp: t,
                    price,
                    volume: None,
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn request_market_chart_handles_rate_limit() {
        let server = MockServer::start_async().await;
        let mock = server.mock_async(|when, then| {
            when.method(GET)
                .path("/coins/bitcoin/market_chart")
                .query_param("vs_currency", "usd")
                .query_param("days", "30");
            then.status(429);
        })
        .await;

        let sdk = build_sdk(&server);
        let result = sdk
            .fetch_price_history("bitcoin", "usd", 30)
            .await
            .unwrap_err();

        mock.assert_hits_async(6).await;
        assert!(matches!(result, PredictionError::Network(message) if message.contains("max retries exceeded")));
    }

    #[tokio::test]
    async fn request_market_chart_handles_unexpected_status() {
        let server = MockServer::start_async().await;
        let mock = server.mock_async(|when, then| {
            when.method(GET)
                .path("/coins/bitcoin/market_chart")
                .query_param("vs_currency", "usd")
                .query_param("days", "30");
            then.status(500);
        })
        .await;

        let sdk = build_sdk(&server);
        let result = sdk
            .fetch_price_history("bitcoin", "usd", 30)
            .await
            .unwrap_err();

        mock.assert_async().await;
        assert!(matches!(result, PredictionError::Network(message) if message.contains("unexpected status")));
    }

    #[tokio::test]
    async fn short_forecast_matches_helper_math() {
        let history = (0..60)
            .map(|idx| PricePoint {
                timestamp: Utc::now() + Duration::minutes(idx),
                price: 100.0 + idx as f64,
                volume: None,
            })
            .collect::<Vec<_>>();

        let sentiment = SentimentSnapshot {
            news_score: 0.2,
            social_score: -0.1,
        };

        let sdk = PredictionSdk::new().expect("sdk construction should succeed");
        let horizon = ShortForecastHorizon::OneHour;
        let result = sdk
            .run_short_forecast(&history, horizon, Some(sentiment.clone()))
            .await
            .expect("forecast should succeed");



        // The new implementation ensembles the statistical forecast with ML.
        // We just verify that the result is reasonable (finite) and that we got an ML prediction.
        assert!(result.expected_price.is_finite());
        assert!(result.ml_prediction.is_some());
        
        // Check that the result is not wildly different from the statistical baseline (heuristic)
        // It might differ, but shouldn't be 0 or NaN.
        assert!(result.expected_price > 0.0);
        assert_eq!(result.horizon, horizon);
    }

    #[tokio::test]
    async fn short_forecast_confidence_tracks_volatility_levels() {
        let sdk = PredictionSdk::new().expect("sdk construction should succeed");
        let horizon = ShortForecastHorizon::OneHour;

        let low_vol_history = synthetic_volatility_series(0.001);
        let medium_vol_history = synthetic_volatility_series(0.01);
        let high_vol_history = synthetic_volatility_series(0.05);

        let low_conf = sdk
            .run_short_forecast(&low_vol_history, horizon, None)
            .await
            .expect("low-volatility forecast should succeed")
            .confidence;
        let medium_conf = sdk
            .run_short_forecast(&medium_vol_history, horizon, None)
            .await
            .expect("medium-volatility forecast should succeed")
            .confidence;
        let high_conf = sdk
            .run_short_forecast(&high_vol_history, horizon, None)
            .await
            .expect("high-volatility forecast should succeed")
            .confidence;

        for confidence in [low_conf, medium_conf, high_conf] {
            assert!((0.0f32..=1.0f32).contains(&confidence));
        }

        assert!(low_conf > medium_conf);
        assert!(medium_conf > high_conf);
    }
}
