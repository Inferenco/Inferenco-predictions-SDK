use prediction_sdk::{
    ForecastHorizon, ForecastRequest, ForecastResponse, LongForecastHorizon, ShortForecastHorizon,
    run_prediction_handler,
};

#[tokio::test]
async fn test_handler_returns_chart_for_short_forecast() {
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
    // When chart is false, it returns ForecastResult directly, not ForecastResponse
    // But wait, let's check the handler implementation.
    // The handler returns `ForecastResponse` ONLY if `chart` is true.
    // If `chart` is false, it returns `ForecastResult`.

    // Attempt to deserialize as ForecastResult
    let result: prediction_sdk::ForecastResult =
        serde_json::from_str(&json).expect("should be ForecastResult");

    // Verify it's NOT a ForecastResponse (which would have a "forecast" field wrapper)
    // Actually, ForecastResponse has "forecast" and "chart" fields.
    // ForecastResult is an enum with "type" and "value".
    // If we try to deserialize as ForecastResponse, it should fail or have missing fields if it was just ForecastResult.
    // But simpler: just assert we got a valid ForecastResult.
    match result {
        prediction_sdk::ForecastResult::Short(_) => {}
        _ => panic!("expected short forecast result"),
    }
}
