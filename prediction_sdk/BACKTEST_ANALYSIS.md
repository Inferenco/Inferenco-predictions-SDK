# Backtest Analysis & Improvement Plan

## 1. Executive Summary
The current backtesting results indicate a fundamental disconnect between the reported **ML Reliability** (>90%) and the actual **Directional Accuracy** (~50%). The trading strategy suffers from this overconfidence, entering trades with "high confidence" that are essentially coin flips, leading to a negative return (-16.13%).

## 2. Performance Analysis

### 2.1 Simple Backtest
- **Metric Mismatch:** The most glaring issue is `Average ML Reliability: 90.4%` vs `Directional Accuracy: 50.2%`.
- **Implication:** The user cannot trust the "Reliability" score. It is not a probability of success; it is currently a measure of "how small the regression error is" (which is naturally small for log returns), not "how likely the direction is correct".
- **Model Bias:** The model appears to be lagging or mean-reverting without capturing trend changes, typical of simple linear regression on time series.

### 2.2 Trading Strategy
- **Losses:** The strategy lost 16.13% while the asset dropped 25.71%. It "outperformed" by losing less, but still lost money.
- **Entry Logic:** The strategy enters trades when `reliability > 50%`. Since reliability is almost always >85%, it enters *every* signal.
- **Risk Management:** The strategy lacks effective stop-loss/take-profit dynamic adjustment based on volatility.

### 2.3 ML Validation
- **Coverage:** 86.8% coverage for prediction intervals is good (target 90%). This suggests the *magnitude* of volatility is being estimated correctly, even if the *direction* is wrong.

## 3. Root Cause Analysis (Codebase Inspection)

### 3.1 Reliability Calculation (`src/analysis.rs`)
The current formula is:
```rust
let reliability = (1.0 / (1.0 + mae * error_scale)).clamp(0.0, 1.0);
```
- **The Flaw:** `mae` is calculated on **log returns**. Log returns are very small numbers (e.g., 0.001). Even with `error_scale` (approx 45), the denominator is `1.0 + 0.045`, resulting in `~0.95`.
- **The Fix:** Reliability must be calibrated to the *scale* of the movement or include a directional accuracy penalty.

### 3.2 Model Architecture
- **Algorithm:** `SVR` with `Kernels::linear()`.
- **Limitation:** Financial time series are highly non-linear. A linear kernel is likely underfitting the complex relationships between RSI, MACD, and Price.
- **Features:** The feature set is decent (RSI, BB Width, Volume), but the linear combination of them is insufficient.

## 4. Improvement Methodology

### Phase 1: Fix Reliability (Calibration)
We need to redefine "Reliability" to be more useful for a user. It should represent "Confidence in the Prediction".
**Proposed Formula:**
Combine `Regression Quality` (current metric) with `Validation Accuracy` (directional).
```rust
// Pseudo-code
let signal_strength = predicted_log_return.abs() / mae;
let reliability = (signal_strength / 2.0).clamp(0.0, 1.0); // If signal is 2x noise, 100% conf.
```
*Rationale:* If the model predicts a 1% move, but its average error is 1%, we should have 0% confidence. If it predicts a 2% move and error is 0.5%, we should have high confidence.

### Phase 2: Enhance Model (SVR Tuning)
- **Kernel:** Switch from `Kernels::linear()` to `Kernels::rbf()` (Radial Basis Function).
- **Hyperparameters:** Tune `C` (Regularization) and `gamma` (RBF width).
- **Feature Scaling:** Ensure all features are strictly normalized (StandardScaler is already implemented, which is good).

### Phase 3: Strategy Optimization
- **Thresholds:** Update the strategy to require a higher *predicted return* threshold, not just reliability.
- **Stop Loss:** Implement a dynamic stop loss based on the `lower_return` (prediction interval lower bound).

## 5. Action Plan
1.  **Modify `src/analysis.rs`**: Implement the new reliability calculation based on Signal-to-Noise Ratio (SNR).
2.  **Modify `src/analysis.rs`**: Change SVR kernel to RBF.
3.  **Re-run Backtests**: Verify if `simple_backtest` shows more realistic reliability and if `trading_strategy` improves.
