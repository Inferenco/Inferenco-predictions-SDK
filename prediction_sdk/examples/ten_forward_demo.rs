use prediction_sdk::ten_forward::{
    Candle, FeaturePoint, MicrostructureFeatures, TechnicalFeatures,
};

fn main() {
    println!("Initializing Ten Forward Data Structures...");

    // Create a dummy candle
    let candle = Candle {
        open_time: 1678886400000,
        close_time: 1678886459999,
        open: 25000.0,
        high: 25100.0,
        low: 24950.0,
        close: 25050.0,
        volume: 150.5,
        quote_volume: 3765000.0,
        n_trades: 500,
        taker_buy_base: 80.0,
        taker_buy_quote: 2004000.0,
    };

    // Create dummy technical features
    let tech = TechnicalFeatures {
        rsi_14: 55.0,
        ema_8: 25040.0,
        ema_13: 25030.0,
        ema_21: 25020.0,
        ema_34: 25000.0,
        ema_55: 24980.0,
        macd: 10.5,
        macd_signal: 9.0,
        macd_hist: 1.5,
        bb_upper: 25200.0,
        bb_middle: 25000.0,
        bb_lower: 24800.0,
        atr_14: 120.0,
    };

    // Create dummy microstructure features
    let micro = MicrostructureFeatures {
        bid_ask_spread: 0.1,
        top5_bid_volume: 10.0,
        top5_ask_volume: 12.0,
        book_imbalance: -0.09,
        avg_trade_size_1m: 0.3,
        trade_imbalance_1m: 5.0,
    };

    // Combine into a feature point
    let point = FeaturePoint {
        candle,
        tech,
        micro,
    };

    println!("\nSuccessfully created FeaturePoint:");
    println!("{:#?}", point);
    println!("\nTen Forward module is correctly linked and accessible!");
}
