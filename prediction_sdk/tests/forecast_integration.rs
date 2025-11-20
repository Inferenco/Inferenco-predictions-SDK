use chrono::{Duration, Utc};
use httpmock::prelude::*;
use prediction_sdk::{ForecastHorizon, LongForecastHorizon, PredictionSdk, ShortForecastHorizon};

fn build_prices(count: usize, start: f64, step: f64) -> Vec<[f64; 2]> {
    let base_time = Utc::now() - Duration::minutes(count as i64);
    (0..count)
        .map(|idx| {
            let ts = (base_time + Duration::minutes(idx as i64)).timestamp_millis() as f64;
            [ts, start + step * idx as f64]
        })
        .collect()
}

#[tokio::test]
async fn forecast_with_fetch_handles_short_horizon() {
    let server = MockServer::start_async().await;
    let prices = build_prices(60, 100.0, 0.5);

    let mock = server
        .mock_async(|when, then| {
            when.method(GET)
                .path("/coins/bitcoin/market_chart/range")
                .query_param("vs_currency", "usd")
                .query_param_exists("from")
                .query_param_exists("to");
            then.status(200)
                .header("content-type", "application/json")
                .json_body_obj(&serde_json::json!({ "prices": prices, "total_volumes": [] }));
        })
        .await;

    let client = reqwest::Client::builder().build().unwrap();
    let sdk = PredictionSdk::with_client(client, Some(server.base_url()));

    let result = sdk
        .forecast_with_fetch(
            "bitcoin",
            "usd",
            ForecastHorizon::Short(ShortForecastHorizon::OneHour),
            None,
        )
        .await
        .expect("short forecast should succeed");

    mock.assert_async().await;
    match result {
        prediction_sdk::ForecastResult::Short(short) => {
            assert_eq!(short.horizon, ShortForecastHorizon::OneHour);
            assert!(short.expected_price.is_finite());
            // Verify advanced features
            assert!(short.technical_signals.is_some());
            assert!(short.ml_prediction.is_some());
            assert!(short.ml_reliability.is_some());
        }
        _ => panic!("expected short forecast"),
    }
}

#[tokio::test]
async fn forecast_with_fetch_handles_long_horizon() {
    let server = MockServer::start_async().await;
    let prices = build_prices(10, 200.0, 1.0);

    let mock = server
        .mock_async(|when, then| {
            when.method(GET)
                .path("/coins/bitcoin/market_chart/range")
                .query_param("vs_currency", "usd")
                .query_param_exists("from")
                .query_param_exists("to");
            then.status(200)
                .header("content-type", "application/json")
                .json_body_obj(&serde_json::json!({ "prices": prices, "total_volumes": [] }));
        })
        .await;

    let client = reqwest::Client::builder().build().unwrap();
    let sdk = PredictionSdk::with_client(client, Some(server.base_url()));

    let result = sdk
        .forecast_with_fetch(
            "bitcoin",
            "usd",
            ForecastHorizon::Long(LongForecastHorizon::OneMonth),
            None,
        )
        .await
        .expect("long forecast should succeed");

    mock.assert_async().await;
    match result {
        prediction_sdk::ForecastResult::Long(long) => {
            assert_eq!(long.horizon, LongForecastHorizon::OneMonth);
            assert!(long.mean_price.is_finite());
            assert!(long.percentile_10 <= long.percentile_90);
        }
        _ => panic!("expected long forecast"),
    }
}
