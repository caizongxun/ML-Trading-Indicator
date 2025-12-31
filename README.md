# ML Trading Indicator - TradingView Pine Script V5

Comprehensive machine learning trading indicator system for TradingView with pending order placement and ML training data generation.

## Overview

ML Order Helper is a complete trading system designed to:
- Generate multiple technical indicators (RSI, MACD, Stochastic, Bollinger Bands)
- Create custom composite indicators for market analysis
- Predict optimal pending order placement levels
- Export standardized features for machine learning model training
- Evaluate order fill probability and profitability

## Repository Contents

1. **ML_Order_Helper_V5.pine** - Pine Script V5 indicator for TradingView
2. **ml_training_script.py** - Python ML model training and prediction system
3. **README.md** - This documentation

## Pine Script Indicator Features

### Base Indicators (5)

- **RSI (Relative Strength Index)** - Momentum oscillator measuring rate of price changes
- **Stochastic Oscillator** - Compares closing price to price range over time period
- **MACD (Moving Average Convergence Divergence)** - Trend-following momentum indicator
- **Bollinger Bands** - Volatility and price levels relative to moving average
- **ATR (Average True Range)** - Volatility measurement for pending order distance

### Custom Indicators (4)

#### 1. Momentum Confluence Score (-100 to +100)
Combines RSI, Stochastic, and MACD signals:
- Values > +50: Strong bullish momentum
- Values < -50: Strong bearish momentum
- -20 to +20: Market neutral

#### 2. Volatility Expansion Index (0 to 100)
Measures Bollinger Bands width relative to historical average:
- > 60: High volatility, suitable for range trading
- < 40: Low volatility, possible breakout brewing

#### 3. RSI Convergence (0 to 100)
Measures RSI deviation from its Bollinger Bands:
- > 80: RSI extreme, potential reversal
- < 20: RSI converged, trend may continue

#### 4. Composite Signal (-100 to +100)
Weighted combination of all custom indicators:
- Unified trading signal combining all analysis

### Pending Order Levels

**Buy Pending Level** = Recent Low - (ATR * 0.5)
- Places buy order below support
- Stop Loss: Buy Pending - ATR
- Take Profit: Recent High (1:2 risk-reward)

**Sell Pending Level** = Recent High + (ATR * 0.5)
- Places sell order above resistance
- Stop Loss: Sell Pending + ATR
- Take Profit: Recent Low (1:2 risk-reward)

## Input Parameters (Corrected Type Declarations)

All Pine Script inputs use proper type specifications:

```pine
rsi_src = input(close, title="RSI Source")
rsi_length = input.int(14, title="RSI Length", minval=2, maxval=100)

bb_src = input(close, title="BB Source")
bb_length = input.int(20, title="BB Length", minval=1, maxval=500)
bb_mult = input.float(2.0, title="BB StdDev", minval=0.1, maxval=5, step=0.1)

macd_fast = input.int(12, title="MACD Fast")
macd_slow = input.int(26, title="MACD Slow")
macd_signal = input.int(9, title="MACD Signal")

stoch_k = input.int(14, title="Stoch %K")
stoch_d = input.int(3, title="Stoch %D")
stoch_smooth = input.int(3, title="Stoch Smooth")

atr_length = input.int(14, title="ATR Length")

sr_lookback = input.int(5, title="S/R Lookback Bars")
sr_sensitivity = input.float(0.3, title="S/R Sensitivity (%)", minval=0.1, maxval=5.0, step=0.1)

show_base_indicators = input.bool(true, title="Show Base Indicators")
show_custom_indicators = input.bool(true, title="Show Custom Indicators")
show_order_levels = input.bool(true, title="Show Pending Order Levels")
show_ml_data = input.bool(true, title="Show ML Data Table")
```

## Installation

### TradingView Setup

1. Open any chart on TradingView
2. Click "Indicators" → "New Indicator"
3. Copy entire content from `ML_Order_Helper_V5.pine`
4. Paste into Pine Script editor
5. Click "Add to Chart"

### Python Setup

```bash
# Install required packages
pip install pandas scikit-learn numpy matplotlib seaborn

# Run training script
python ml_training_script.py
```

## ML Data Export Format

The indicator displays a real-time data table with the following structure:

### Base Indicators
- RSI
- Stochastic %K
- MACD
- Bollinger Bands Width

### Custom Indicators
- Momentum Confluence Score
- Volatility Expansion Index
- RSI Convergence
- Composite Signal

### Order Levels
- Buy Pending Level
- Sell Pending Level
- Buy Stop Loss
- Sell Stop Loss

## Machine Learning Training

### Data Preparation

1. Export indicator data from TradingView as CSV
2. Format: `datetime, rsi, stoch, macd, bb_width, momentum_score, volatility_index, rsi_convergence, composite_signal, close_price, buy_pending_level, sell_pending_level, order_filled, order_profitable`

### Training Models

The Python script trains 3 models:

**Model 1: Order Fill Classifier**
- Predicts probability of pending order being triggered
- Algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Automatically selects best performer

**Model 2: Order Profit Classifier**
- Predicts probability of order being profitable
- Same algorithms with automatic selection

**Model 3: Pending Level Regressor**
- Predicts optimal buy/sell pending order prices
- Separate models for buy and sell

### Usage Example

```python
from ml_training_script import MLDataHandler, MLModelTrainer, OrderPredictor

# Load and prepare data
handler = MLDataHandler()
handler.load_data('trading_data.csv')
handler.preprocess_data()
handler.feature_engineering()

X, y, feature_cols = handler.prepare_ml_data()

# Train models
trainer = MLModelTrainer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
trainer.train_pending_level_regressor(X_train, y_train, X_test, y_test)

# Make predictions
predictor = OrderPredictor(trainer, feature_cols)
current_features = {'rsi': 45.2, 'stoch': 62.1, ...}
prediction = predictor.predict_order_signal(current_features)
print(prediction['recommendation'])  # 'STRONG_BUY', 'BUY', 'WATCH', 'HOLD'
```

## Recommended Parameter Settings

### Short-term Trading (5-15 minutes)
```
RSI Length: 7-9
BB Length: 10-12
S/R Lookback: 3-5 bars
ATR Length: 7-10
```

### Medium-term Trading (1-4 hours)
```
RSI Length: 12-14 (default)
BB Length: 18-22 (default)
S/R Lookback: 5-8 bars
ATR Length: 13-15 (default)
```

### Long-term Trading (Daily and above)
```
RSI Length: 14-21
BB Length: 20-30
S/R Lookback: 8-12 bars
ATR Length: 14-20
```

## Trading Signals

### Buy Setup
- Composite Signal < -50
- RSI < 30 (oversold)
- Stochastic %K < 20 (oversold)
- Price approaching Buy Pending Level

### Sell Setup
- Composite Signal > +50
- RSI > 70 (overbought)
- Stochastic %K > 80 (overbought)
- Price approaching Sell Pending Level

## Model Evaluation

After training, the system outputs:

- Training accuracy for each model
- Test accuracy for each model
- ROC-AUC scores for classification models
- R-squared and MAE for regression models
- Saved models in `./models/` directory

## Important Notes

1. **Type Declarations**: All `input()` functions now use proper type specifications (`input.int()`, `input.float()`, `input.bool()`)
2. **Data Quality**: Minimum 2000 candles recommended for model training
3. **Retraining**: Update models monthly or quarterly as market conditions change
4. **Risk Management**: Always use stop losses and position sizing
5. **Paper Trading**: Validate strategy with paper trading before live trading

## GitHub Repository

All files available at: [ML-Trading-Indicator](https://github.com/caizongxun/ML-Trading-Indicator)

## Version History

- **v1.0** (2025-12-31): Initial release with corrected input type declarations

## License

Open source for educational and personal trading use.

## Support

For issues or suggestions, please open an issue in the GitHub repository.
