# Work Summary

## Chart-aware forecast requests and responses
- Added a `chart` flag to `ForecastRequest` so callers can opt into chart output without breaking existing consumers, and introduced `ForecastResponse` to bundle the forecast with optional chart data.
- Updated the prediction handler to propagate the flag, fetch chart-ready candles alongside price history, and emit the combined response shape when requested.

## Monte Carlo path sampling for chart projections
- Extended Monte Carlo simulations to optionally record per-day price distributions, aggregating percentile bands so long-horizon forecasts can return a time-series projection instead of just terminal values.
- Long-horizon forecasts now translate the sampled steps into dated `ForecastBandPoint` entries aligned with the last historical timestamp for straightforward charting.

## Chart-ready historical candles
- Added CoinGecko OHLC fetching (with rate-limit handling) and fallback aggregation from raw price history to produce `ChartCandle` structures suitable for candlestick rendering.
- Chart responses include both the historical candles and, when available, the projected band so clients can plot past and future prices together.
