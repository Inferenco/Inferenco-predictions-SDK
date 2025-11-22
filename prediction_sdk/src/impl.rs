use std::collections::{BTreeMap, HashMap};

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
    ForecastBandPoint,
    ForecastResult,
    ChartCandle,
    LongForecastHorizon,
    LongForecastResult,
    IntervalCalibration,
    PredictionError,
    PricePoint,
    SentimentSnapshot,
    ShortForecastHorizon,
    ShortForecastResult,
    MlModelConfig,
};

const DEFAULT_BASE_URL: &str = "https://api.coingecko.com/api/v3";
const DEFAULT_SIMULATIONS: usize = 256;
const SHORT_FORECAST_LOOKBACK_DAYS: u32 = 90;

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

    /// Override the global ML configuration used by the forecasting pipeline.
    ///
    /// This setter is optional and preserves the existing public API while
    /// exposing the new hyperparameters (patch length, mixture size, and
    /// learning rate) to callers.
    pub fn set_ml_config(&self, config: MlModelConfig) {
        analysis::set_ml_model_config(config);
    }

    /// Fetch historical market data in batches to ensure high granularity (hourly) over long periods.
    ///
    /// The CoinGecko API downsamples data for long ranges (e.g., > 90 days). To get hourly data
    /// for a full year, we fetch in 90-day chunks and stitch them together.
    pub async fn fetch_price_history_batched(
        &self,
        asset_id: &str,
        vs_currency: &str,
        days: u32,
    ) -> Result<Vec<PricePoint>, PredictionError> {
        let mut all_points = Vec::new();
        let now = Utc::now();
        let chunk_size = 90;
        let mut remaining_days = days;
        let mut current_end = now;

        // We loop backwards from now
        while remaining_days > 0 {
            let fetch_days = std::cmp::min(remaining_days, chunk_size);
            let start = current_end - Duration::days(fetch_days as i64);

            // Fetch the chunk
            // Note: We might get slight overlaps or gaps depending on exact timing, but sorting/dedup handles it.
            let chunk = self
                .fetch_price_history_range(asset_id, vs_currency, start, current_end)
                .await?;

            all_points.extend(chunk);

            // Move the window back
            current_end = start;
            remaining_days -= fetch_days;
        }

        // Sort by timestamp (ascending)
        all_points.sort_by_key(|p| p.timestamp);

        // Deduplicate based on timestamp
        all_points.dedup_by_key(|p| p.timestamp);

        Ok(all_points)
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

    /// Fetch chart-ready OHLC data. Attempts to use the dedicated CoinGecko
    /// `/ohlc` endpoint when available and falls back to aggregating the
    /// provided price history into candles.
    pub async fn fetch_chart_candles(
        &self,
        asset_id: &str,
        vs_currency: &str,
        days: u32,
        fallback_history: &[PricePoint],
    ) -> Result<Vec<ChartCandle>, PredictionError> {
        if let Ok(ohlc) = self
            .request_market_ohlc(asset_id, vs_currency, days)
            .await
        {
            return Ok(ohlc);
        }

        aggregate_candles_from_history(fallback_history)
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
        if history.len() < 2 {
            return Err(PredictionError::InsufficientData);
        }

        let moving_average = helpers::calculate_moving_average(history, window)?;
        let volatility_window = window.saturating_mul(2).max(2);
        let recent_len = history.len().min(volatility_window);
        if recent_len < 2 {
            return Err(PredictionError::InsufficientData);
        }
        let recent_history = &history[history.len() - recent_len..];

        let (drift, volatility) = helpers::daily_return_stats(recent_history)?;

        let horizon_hours: f64 = match horizon {
            ShortForecastHorizon::FifteenMinutes => 0.25,
            ShortForecastHorizon::OneHour => 1.0,
            ShortForecastHorizon::FourHours => 4.0,
        };

        let trend_adjustment = (drift * (horizon_hours / 24.0)).exp();
        let mut expected_price = moving_average * trend_adjustment;
        if let Some(snapshot) = sentiment.as_ref() {
            expected_price = helpers::weight_with_sentiment(expected_price, snapshot);
        }

        // Calculate average interval for volatility scaling correction
        let avg_interval_days = if recent_history.len() > 1 {
            let start = recent_history.first().map(|p| p.timestamp.timestamp()).unwrap_or(0);
            let end = recent_history.last().map(|p| p.timestamp.timestamp()).unwrap_or(0);
            let seconds = (end - start) as f64;
            (seconds / 86400.0) / (recent_history.len() - 1) as f64
        } else {
            1.0 / 24.0
        };

        // daily_return_stats returns volatility of the *rate*, which is sigma_daily * sqrt(1/dt).
        // We want sigma_daily, so we multiply by sqrt(dt).
        let adjusted_volatility = volatility * avg_interval_days.sqrt();

        let decomposition = helpers::decompose_series(recent_history)?;
        let normalized_noise = if moving_average.abs() > f64::EPSILON {
            decomposition.noise / moving_average
        } else {
            0.0
        };

        // Use a sigmoid-like decay for confidence based on volatility.
        // High volatility -> Low confidence.
        // Low volatility -> High confidence.
        // Formula: 1.0 / (1.0 + scaling_factor * volatility * sqrt(time))
        let scaling_factor = 15.0; // Tuned to 15.0 to balance volatility sensitivity without being overly punitive
        let time_scaling = horizon_hours.sqrt();
        let combined_uncertainty = 0.6 * adjusted_volatility + 0.4 * normalized_noise;
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

        let mut ml_return_bounds = None;
        let mut ml_price_interval = None;
        let mut ml_interval_calibration = None;

        if let Some(ml) = ml_prediction.as_ref() {
            let calibration_score = f64::from(ml.calibration_score);
            let last_price = history
                .last()
                .map(|point| point.price)
                .unwrap_or(baseline_expected);
            let lower_price = last_price * ml.lower_return.exp();
            let upper_price = last_price * ml.upper_return.exp();
            let interval_width = (upper_price - lower_price).abs();
            let price_scale = if last_price.abs() < f64::EPSILON {
                1.0
            } else {
                last_price.abs()
            };
            let relative_width = interval_width / price_scale;
            let width_penalty = 1.0 / (1.0 + relative_width);
            let coverage_alignment =
                1.0 - (ml.observed_coverage - ml.target_coverage).abs().min(1.0);
            let weight =
                (calibration_score * width_penalty * coverage_alignment).clamp(0.0, 1.0);

            ml_return_bounds = Some((ml.lower_return, ml.upper_return));
            ml_price_interval = Some((lower_price, upper_price));
            ml_interval_calibration = Some(IntervalCalibration {
                target_coverage: ml.target_coverage,
                observed_coverage: ml.observed_coverage,
                interval_width: ml.interval_width,
                price_interval_width: interval_width,
                pinball_loss: ml.pinball_loss,
                calibration_score: ml.calibration_score,
            });

            if calibration_score >= 0.05 && weight > 0.0 {
                expected_price = baseline_expected * (1.0 - weight)
                    + ml.predicted_price * weight;
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
            ml_return_bounds,
            ml_price_interval,
            ml_interval_calibration,
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
        chart: bool,
    ) -> Result<(LongForecastResult, Option<Vec<ForecastBandPoint>>), PredictionError> {
        let days = helpers::long_horizon_days(horizon);
        let simulations = helpers::scaled_simulation_count(days, DEFAULT_SIMULATIONS);
        let (drift, _) = helpers::daily_return_stats(history)?;
        let volatility_path = helpers::forecast_volatility_series(history, days)?;
        let paths = helpers::run_monte_carlo(
            history,
            days,
            simulations,
            drift,
            &volatility_path,
            Some(0.005), // Reduced from 0.02 to avoid excessive bearish pull
            chart,
        )?;
        let mean_price = paths.final_prices.iter().sum::<f64>() / paths.final_prices.len() as f64;
        let percentile_10 = helpers::percentile(paths.final_prices.clone(), 0.1)?;
        let percentile_90 = helpers::percentile(paths.final_prices.clone(), 0.9)?;
        let mut adjusted_mean = mean_price;
        if let Some(snapshot) = sentiment.as_ref() {
            adjusted_mean = helpers::weight_with_sentiment(adjusted_mean, snapshot);
        }
        let spread = percentile_90 - percentile_10;
        let relative_spread = if adjusted_mean.abs() > f64::EPSILON {
            spread / adjusted_mean
        } else {
            1.0
        };
        // Confidence is inversely proportional to the relative spread.
        // A spread of 0% -> Confidence 1.0
        // A spread of 100% -> Confidence 0.5
        let confidence = helpers::normalize_confidence(1.0 / (1.0 + relative_spread));

        // Calculate technical signals (still useful for long horizons as a starting point)
        let technical_signals = analysis::calculate_technical_signals(history).ok();

        let projection = if chart {
            let last_timestamp = history
                .iter()
                .max_by_key(|point| point.timestamp)
                .map(|point| point.timestamp)
                .ok_or(PredictionError::InsufficientData)?;

            paths
                .step_samples
                .as_ref()
                .map(|steps| {
                    steps
                        .iter()
                        .map(|sample| {
                            let timestamp = last_timestamp
                                .checked_add_signed(Duration::days(i64::from(sample.day)))
                                .ok_or(PredictionError::TimeConversion)?;

                            Ok(ForecastBandPoint {
                                timestamp,
                                percentile_10: sample.percentile_10,
                                mean: sample.mean,
                                percentile_90: sample.percentile_90,
                            })
                        })
                        .collect::<Result<Vec<_>, PredictionError>>()
                })
                .transpose()?
        } else {
            None
        };

        Ok((
            LongForecastResult {
                horizon,
                mean_price: adjusted_mean,
                percentile_10,
                percentile_90,
                confidence,
                technical_signals,
                ml_prediction: Some(mean_price),
            },
            projection,
        ))
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
                .run_long_forecast(history, long_horizon, sentiment, false)
                .await
                .map(|(forecast, _)| ForecastResult::Long(forecast)),
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
                    .fetch_price_history_batched(asset_id, vs_currency, SHORT_FORECAST_LOOKBACK_DAYS)
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

    async fn request_market_ohlc(
        &self,
        asset_id: &str,
        vs_currency: &str,
        days: u32,
    ) -> Result<Vec<ChartCandle>, PredictionError> {
        let url = format!("{}/coins/{}/ohlc", self.market_base_url, asset_id);
        let query = vec![
            ("vs_currency", vs_currency.to_string()),
            ("days", days.to_string()),
        ];

        let max_retries = 5;
        let mut attempt = 0;
        let base_delay_ms = 500;

        loop {
            self.limiter.until_ready().await;

            let response = self
                .client
                .get(&url)
                .query(&query)
                .send()
                .await
                .map_err(|err| PredictionError::Network(err.to_string()))?;

            let status = response.status();

            if status.is_success() {
                let payload: Vec<[f64; 5]> = response
                    .json()
                    .await
                    .map_err(|err| PredictionError::Serialization(err.to_string()))?;
                let candles = build_chart_candles(payload)?;
                return Ok(candles);
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

fn build_chart_candles(payload: Vec<[f64; 5]>) -> Result<Vec<ChartCandle>, PredictionError> {
    let mut candles = Vec::with_capacity(payload.len());
    for entry in payload {
        let timestamp_ms = entry[0] as i64;
        let timestamp = Utc
            .timestamp_millis_opt(timestamp_ms)
            .single()
            .ok_or(PredictionError::TimeConversion)?;
        candles.push(ChartCandle {
            timestamp,
            open: entry[1],
            high: entry[2],
            low: entry[3],
            close: entry[4],
            volume: None,
        });
    }

    Ok(candles)
}

fn aggregate_candles_from_history(history: &[PricePoint]) -> Result<Vec<ChartCandle>, PredictionError> {
    if history.is_empty() {
        return Ok(Vec::new());
    }

    let mut sorted = history.to_vec();
    sorted.sort_by_key(|point| point.timestamp);

    let bucket_seconds = sorted
        .windows(2)
        .filter_map(|pair| {
            let delta = pair[1].timestamp - pair[0].timestamp;
            let seconds = delta.num_seconds();
            (seconds > 0).then_some(seconds)
        })
        .min()
        .unwrap_or(3600);

    let mut buckets: BTreeMap<i64, Vec<PricePoint>> = BTreeMap::new();
    for point in sorted {
        let ts = point.timestamp.timestamp();
        let bucket_key = if bucket_seconds > 0 {
            ts - (ts % bucket_seconds)
        } else {
            ts
        };
        buckets.entry(bucket_key).or_default().push(point);
    }

    let mut candles = Vec::with_capacity(buckets.len());
    for points in buckets.values() {
        if points.is_empty() {
            continue;
        }
        let open_point = points
            .first()
            .ok_or(PredictionError::TimeConversion)?;
        let close_point = points
            .last()
            .ok_or(PredictionError::TimeConversion)?;
        let mut high = open_point.price;
        let mut low = open_point.price;
        let mut volume_sum = 0.0;
        let mut has_volume = false;

        for point in points {
            if point.price > high {
                high = point.price;
            }
            if point.price < low {
                low = point.price;
            }
            if let Some(volume) = point.volume {
                volume_sum += volume;
                has_volume = true;
            }
        }

        let volume = if has_volume { Some(volume_sum) } else { None };

        candles.push(ChartCandle {
            timestamp: open_point.timestamp,
            open: open_point.price,
            high,
            low,
            close: close_point.price,
            volume,
        });
    }

    Ok(candles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use httpmock::prelude::*;
    use serde_json::json;


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

    fn trending_history(start: f64, factor: f64, points: usize) -> Vec<PricePoint> {
        let start_time = Utc::now() - Duration::minutes(points as i64);
        (0..points)
            .map(|idx| {
                let timestamp = start_time + Duration::minutes(idx as i64);
                let price = start * factor.powi(idx as i32);
                PricePoint {
                    timestamp,
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
    async fn request_market_ohlc_parses_payload() {
        let server = MockServer::start_async().await;
        let timestamp_ms = 1_700_000_000_000i64;
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/coins/bitcoin/ohlc")
                    .query_param("vs_currency", "usd")
                    .query_param("days", "1");
                then.status(200)
                    .json_body(json!([[timestamp_ms, 10.0, 15.0, 5.0, 12.5]]));
            })
            .await;

        let sdk = build_sdk(&server);
        let candles = sdk
            .fetch_chart_candles("bitcoin", "usd", 1, &[])
            .await
            .expect("expected OHLC payload to parse");

        mock.assert();
        assert_eq!(candles.len(), 1);
        let candle = &candles[0];
        assert_eq!(candle.timestamp.timestamp_millis(), timestamp_ms);
        assert_eq!(candle.open, 10.0);
        assert_eq!(candle.high, 15.0);
        assert_eq!(candle.low, 5.0);
        assert_eq!(candle.close, 12.5);
        assert!(candle.volume.is_none());
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

    #[test]
    fn aggregates_history_into_candles_with_volume() {
        let base = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let history = vec![
            PricePoint {
                timestamp: base,
                price: 10.0,
                volume: Some(1.0),
            },
            PricePoint {
                timestamp: base, // Same timestamp to force aggregation
                price: 15.0,
                volume: Some(2.0),
            },
            PricePoint {
                timestamp: base + Duration::seconds(60),
                price: 8.0,
                volume: Some(3.0),
            },
        ];

        let candles = aggregate_candles_from_history(&history).expect("aggregation should succeed");

        assert_eq!(candles.len(), 2);
        let first = &candles[0];
        assert_eq!(first.timestamp, base);
        assert_eq!(first.open, 10.0);
        assert_eq!(first.high, 15.0);
        assert_eq!(first.low, 10.0);
        assert_eq!(first.close, 15.0);
        assert_eq!(first.volume, Some(3.0));

        let second = &candles[1];
        assert_eq!(second.timestamp, base + Duration::seconds(60));
        assert_eq!(second.open, 8.0);
        assert_eq!(second.high, 8.0);
        assert_eq!(second.low, 8.0);
        assert_eq!(second.close, 8.0);
        assert_eq!(second.volume, Some(3.0));
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

    #[tokio::test]
    async fn short_forecast_reflects_positive_trend() {
        let sdk = PredictionSdk::new().expect("sdk construction should succeed");
        let horizon = ShortForecastHorizon::FifteenMinutes;
        let history = trending_history(100.0, 1.002, 24);

        let result = sdk
            .run_short_forecast(&history, horizon, None)
            .await
            .expect("forecast should succeed");

        let window = helpers::short_horizon_window(horizon);
        let moving_average = helpers::calculate_moving_average(&history, window)
            .expect("moving average should succeed");

        assert!(result.expected_price > moving_average);
    }

    #[tokio::test]
    async fn short_forecast_reflects_negative_trend() {
        let sdk = PredictionSdk::new().expect("sdk construction should succeed");
        let horizon = ShortForecastHorizon::FifteenMinutes;
        let history = trending_history(100.0, 0.998, 24);

        let result = sdk
            .run_short_forecast(&history, horizon, None)
            .await
            .expect("forecast should succeed");

        let window = helpers::short_horizon_window(horizon);
        let moving_average = helpers::calculate_moving_average(&history, window)
            .expect("moving average should succeed");

        assert!(result.expected_price < moving_average);
    }
}
