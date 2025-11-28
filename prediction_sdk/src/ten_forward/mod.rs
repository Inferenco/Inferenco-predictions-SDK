pub mod data;
pub mod features;
pub mod ml;

#[derive(Clone, Debug)]
pub struct Candle {
    pub open_time: i64,
    pub close_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub n_trades: u64,
    pub taker_buy_base: f64,
    pub taker_buy_quote: f64,
}

#[derive(Clone, Debug)]
pub struct OrderBookSlice {
    pub time: i64,
    pub bids: Vec<(f64, f64)>, // (price, qty)
    pub asks: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct MicrostructureFeatures {
    pub bid_ask_spread: f64,
    pub top5_bid_volume: f64,
    pub top5_ask_volume: f64,
    pub book_imbalance: f64, // (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub avg_trade_size_1m: f64,
    pub trade_imbalance_1m: f64, // buy_vol - sell_vol
                                 // etc
}

#[derive(Clone, Debug)]
pub struct TechnicalFeatures {
    pub rsi_14: f64,
    pub ema_8: f64,
    pub ema_13: f64,
    pub ema_21: f64,
    pub ema_34: f64,
    pub ema_55: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub bb_upper: f64,
    pub bb_middle: f64,
    pub bb_lower: f64,
    pub atr_14: f64,
    // etc
}

#[derive(Clone, Debug)]
pub struct FeaturePoint {
    pub candle: Candle,
    pub tech: TechnicalFeatures,
    pub micro: MicrostructureFeatures,
    // optional time-of-day / day-of-week encodings, etc.
}
