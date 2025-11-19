use moka::Expiry;
use moka::future::Cache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use crate::dto::{
    LongForecastHorizon, LongForecastResult, PricePoint, SentimentSnapshot, ShortForecastHorizon,
    ShortForecastResult,
};

/// Cache key for API responses (CoinGecko market data)
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ApiCacheKey {
    DaysLookback {
        asset_id: String,
        vs_currency: String,
        days: u32,
    },
    DateRange {
        asset_id: String,
        vs_currency: String,
        from_ts: i64,
        to_ts: i64,
    },
}

impl ApiCacheKey {
    fn calculate_ttl(&self) -> Duration {
        let days = match self {
            ApiCacheKey::DaysLookback { days, .. } => *days,
            ApiCacheKey::DateRange { from_ts, to_ts, .. } => {
                let diff_secs = to_ts - from_ts;
                (diff_secs / 86400) as u32
            }
        };

        // TTL strategy based on data window
        if days <= 7 {
            Duration::from_secs(5 * 60) // 5 minutes
        } else if days <= 30 {
            Duration::from_secs(15 * 60) // 15 minutes
        } else {
            Duration::from_secs(60 * 60) // 1 hour
        }
    }
}

/// Cache key for forecast results
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ForecastCacheKey {
    Short {
        asset_id: String,
        vs_currency: String,
        horizon: ShortForecastHorizon,
        sentiment_hash: u64,
    },
    Long {
        asset_id: String,
        vs_currency: String,
        horizon: LongForecastHorizon,
        sentiment_hash: u64,
    },
}

impl ForecastCacheKey {
    fn calculate_ttl(&self) -> Duration {
        match self {
            ForecastCacheKey::Short { horizon, .. } => match horizon {
                ShortForecastHorizon::FifteenMinutes => Duration::from_secs(3 * 60), // 3 minutes
                ShortForecastHorizon::OneHour => Duration::from_secs(10 * 60),       // 10 minutes
                ShortForecastHorizon::FourHours => Duration::from_secs(30 * 60),     // 30 minutes
            },
            ForecastCacheKey::Long { horizon, .. } => match horizon {
                LongForecastHorizon::OneDay => Duration::from_secs(2 * 60 * 60), // 2 hours
                LongForecastHorizon::ThreeDays => Duration::from_secs(4 * 60 * 60), // 4 hours
                LongForecastHorizon::OneWeek => Duration::from_secs(6 * 60 * 60), // 6 hours
                LongForecastHorizon::OneMonth => Duration::from_secs(12 * 60 * 60), // 12 hours
                LongForecastHorizon::ThreeMonths => Duration::from_secs(12 * 60 * 60), // 12 hours
                LongForecastHorizon::SixMonths => Duration::from_secs(12 * 60 * 60), // 12 hours
                LongForecastHorizon::OneYear => Duration::from_secs(12 * 60 * 60), // 12 hours
                LongForecastHorizon::FourYears => Duration::from_secs(12 * 60 * 60), // 12 hours
            },
        }
    }
}

/// Custom expiry for API cache based on key
struct ApiCacheExpiry;

impl Expiry<ApiCacheKey, Vec<PricePoint>> for ApiCacheExpiry {
    fn expire_after_create(
        &self,
        key: &ApiCacheKey,
        _value: &Vec<PricePoint>,
        _current_time: std::time::Instant,
    ) -> Option<Duration> {
        Some(key.calculate_ttl())
    }
}

/// Custom expiry for forecast cache based on key
struct ForecastCacheExpiry;

#[derive(Debug, Clone)]
pub enum ForecastCacheValue {
    Short(ShortForecastResult),
    Long(LongForecastResult),
}

impl Expiry<ForecastCacheKey, ForecastCacheValue> for ForecastCacheExpiry {
    fn expire_after_create(
        &self,
        key: &ForecastCacheKey,
        _value: &ForecastCacheValue,
        _current_time: std::time::Instant,
    ) -> Option<Duration> {
        Some(key.calculate_ttl())
    }
}

/// Cache for CoinGecko API responses
pub struct ApiCache {
    cache: Cache<ApiCacheKey, Vec<PricePoint>>,
}

impl ApiCache {
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(1000)
                .expire_after(ApiCacheExpiry)
                .build(),
        }
    }

    pub async fn get(&self, key: &ApiCacheKey) -> Option<Vec<PricePoint>> {
        self.cache.get(key).await
    }

    pub async fn insert(&self, key: ApiCacheKey, value: Vec<PricePoint>) {
        self.cache.insert(key, value).await;
    }
}

impl Default for ApiCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for forecast results
pub struct ForecastCache {
    cache: Cache<ForecastCacheKey, ForecastCacheValue>,
}

impl ForecastCache {
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(500)
                .expire_after(ForecastCacheExpiry)
                .build(),
        }
    }

    pub async fn get(&self, key: &ForecastCacheKey) -> Option<ForecastCacheValue> {
        self.cache.get(key).await
    }

    pub async fn insert(&self, key: ForecastCacheKey, value: ForecastCacheValue) {
        self.cache.insert(key, value).await;
    }
}

impl Default for ForecastCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash a SentimentSnapshot for cache key
pub fn hash_sentiment(sentiment: &Option<SentimentSnapshot>) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Some(s) = sentiment {
        // Convert floats to bits for hashing
        s.news_score.to_bits().hash(&mut hasher);
        s.social_score.to_bits().hash(&mut hasher);
    } else {
        0u64.hash(&mut hasher);
    }
    hasher.finish()
}
