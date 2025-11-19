//! Entry point for the prediction SDK crate.
//! Consumers should import exported types via the crate root.
//!
//! # Example
//!
//! ```no_run
//! use chrono::{Duration, Utc};
//! use prediction_sdk::{
//!     ForecastHorizon, PredictionSdk, PricePoint, SentimentSnapshot, ShortForecastHorizon,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), prediction_sdk::PredictionError> {
//!     let sdk = PredictionSdk::new()?;
//!
//!     let history: Vec<PricePoint> = (0..50)
//!         .map(|idx| PricePoint {
//!             timestamp: Utc::now() - Duration::minutes(idx),
//!             price: 100.0 + idx as f64,
//!             volume: None,
//!         })
//!         .collect();
//!
//!     let sentiment = SentimentSnapshot {
//!         news_score: 0.25,
//!         social_score: 0.1,
//!     };
//!
//!     let forecast = sdk
//!         .forecast(
//!             &history,
//!             ForecastHorizon::Short(ShortForecastHorizon::OneHour),
//!             Some(sentiment),
//!         )
//!         .await?;
//!
//!     println!("{forecast:?}");
//!     Ok(())
//! }
//! ```

mod r#mod;

pub use r#mod::*;
