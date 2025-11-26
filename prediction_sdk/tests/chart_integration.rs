use httpmock::prelude::*;
use prediction_sdk::{
    ForecastHorizon, ForecastRequest, ForecastResponse, LongForecastHorizon, ShortForecastHorizon,
    run_prediction_handler,
};
use std::sync::OnceLock;

static MOCK_SERVER: OnceLock<MockServer> = OnceLock::new();

fn get_mock_server() -> &'static MockServer {
    MOCK_SERVER.get_or_init(|| {
        let server = MockServer::start();
        unsafe {
            std::env::set_var("COINGECKO_API_URL", server.base_url());
        }
        server
    })
}

#[tokio::test]
async fn test_handler_returns_chart_for_short_forecast() {
    let server = get_mock_server();

    // Mock market chart response with sufficient data (7 days of hourly data = ~168 points)
    let mut prices = Vec::new();
    let start_ts = 1700000000000i64;
    for i in 0..170 {
        prices.push(format!(
            "[{}, {}]",
            start_ts + i * 3600000,
            50000.0 + (i as f64) * 10.0
        ));
    }
    let market_chart_response = format!(
        r#"{{ "prices": [{}], "total_volumes": [] }}"#,
        prices.join(",")
    );

    let _m1 = server.mock(|when, then| {
        when.method(GET)
            .path("/coins/bitcoin/market_chart")
            .query_param("vs_currency", "usd")
            .query_param("days", "7");
        then.status(200)
            .header("content-type", "application/json")
            .body(market_chart_response);
    });

    // Mock OHLC response (optional, but good to have if the code path tries it)
    // The code tries OHLC first for chart candles.
    let ohlc_response = r#"[[1700000000000, 50000.0, 50200.0, 49900.0, 50100.0]]"#;
    let _m2 = server.mock(|when, then| {
        when.method(GET)
            .path("/coins/bitcoin/ohlc")
            .query_param("vs_currency", "usd")
            .query_param("days", "7");
        then.status(200)
            .header("content-type", "application/json")
            .body(ohlc_response);
    });

    let request = ForecastRequest {
        asset_id: "bitcoin".to_string(),
        vs_currency: "usd".to_string(),
        horizon: ForecastHorizon::Short(ShortForecastHorizon::OneHour),
        sentiment: None,
        chart: true,
    };

    let json = run_prediction_handler(request)
        .await
        .expect("handler failed");
    let response: ForecastResponse = serde_json::from_str(&json).expect("deserialization failed");

    assert!(response.chart.is_some(), "chart data should be present");
    let chart = response.chart.unwrap();
    assert!(
        !chart.history.is_empty(),
        "history candles should not be empty"
    );
    assert!(
        chart.projection.is_none(),
        "short forecast should not have projection"
    );
}

#[tokio::test]
async fn test_handler_returns_chart_and_projection_for_long_forecast() {
    let server = get_mock_server();

    // Mock market chart range response for long forecast
    // We need enough data points for the Monte Carlo simulation
    let mut prices = Vec::new();
    let start_ts = 1700000000000i64;
    for i in 0..100 {
        prices.push(format!(
            "[{}, {}]",
            start_ts + i * 86400000,
            2000.0 + (i as f64)
        ));
    }
    let market_chart_range_response = format!(
        r#"{{ "prices": [{}], "total_volumes": [] }}"#,
        prices.join(",")
    );

    let _m1 = server.mock(|when, then| {
        when.method(GET)
            .path("/coins/ethereum/market_chart/range")
            .query_param("vs_currency", "usd");
        then.status(200)
            .header("content-type", "application/json")
            .body(market_chart_range_response);
    });

    // Mock OHLC for the chart data
    let ohlc_response = r#"[[1700000000000, 2000.0, 2050.0, 1950.0, 2010.0]]"#;
    let _m2 = server.mock(|when, then| {
        when.method(GET)
            .path("/coins/ethereum/ohlc")
            .query_param("vs_currency", "usd");
        then.status(200)
            .header("content-type", "application/json")
            .body(ohlc_response);
    });

    let request = ForecastRequest {
        asset_id: "ethereum".to_string(),
        vs_currency: "usd".to_string(),
        horizon: ForecastHorizon::Long(LongForecastHorizon::OneMonth),
        sentiment: None,
        chart: true,
    };

    let json = run_prediction_handler(request)
        .await
        .expect("handler failed");
    let response: ForecastResponse = serde_json::from_str(&json).expect("deserialization failed");

    assert!(response.chart.is_some(), "chart data should be present");
    let chart = response.chart.unwrap();
    assert!(
        !chart.history.is_empty(),
        "history candles should not be empty"
    );
    assert!(
        chart.projection.is_some(),
        "long forecast should have projection"
    );
    let projection = chart.projection.unwrap();
    assert!(
        !projection.is_empty(),
        "projection bands should not be empty"
    );
}

#[tokio::test]
async fn test_handler_omits_chart_when_flag_is_false() {
    let server = get_mock_server();

    let mut prices = Vec::new();
    let start_ts = 1700000000000i64;
    for i in 0..170 {
        prices.push(format!(
            "[{}, {}]",
            start_ts + i * 3600000,
            50000.0 + (i as f64) * 10.0
        ));
    }
    let market_chart_response = format!(
        r#"{{ "prices": [{}], "total_volumes": [] }}"#,
        prices.join(",")
    );

    let _m1 = server.mock(|when, then| {
        when.method(GET)
            .path("/coins/bitcoin/market_chart")
            .query_param("vs_currency", "usd")
            .query_param("days", "7");
        then.status(200)
            .header("content-type", "application/json")
            .body(market_chart_response);
    });

    let request = ForecastRequest {
        asset_id: "bitcoin".to_string(),
        vs_currency: "usd".to_string(),
        horizon: ForecastHorizon::Short(ShortForecastHorizon::OneHour),
        sentiment: None,
        chart: false,
    };

    let json = run_prediction_handler(request)
        .await
        .expect("handler failed");

    // Attempt to deserialize as ForecastResult
    let result: prediction_sdk::ForecastResult =
        serde_json::from_str(&json).expect("should be ForecastResult");

    match result {
        prediction_sdk::ForecastResult::Short(_) => {}
        _ => panic!("expected short forecast result"),
    }
}
