# ML Order Helper - Complete Usage Guide

## System Architecture

The ML Trading Indicator system consists of three main components:

1. **Pine Script V5 Indicator** - Real-time market analysis and order level generation
2. **Data Export Module** - ML training data standardization
3. **Python ML System** - Model training and prediction

## Part 1: TradingView Setup

### Step 1: Add Indicator to Chart

1. Open TradingView in web browser
2. Select any trading pair (EURUSD, BTCUSD, etc.)
3. Click "Indicators" in toolbar
4. Select "New Indicator"
5. Copy entire Pine Script code from `ML_Order_Helper_V5.pine`
6. Paste into script editor
7. Click "Add to Chart"

### Step 2: Configure Parameters

The indicator provides the following configurable parameters:

#### RSI Settings
- **RSI Source**: Select price source (default: close)
- **RSI Length**: Period for RSI calculation (default: 14, range: 2-100)

#### Bollinger Bands Settings
- **BB Source**: Price source for bands (default: close)
- **BB Length**: Moving average period (default: 20, range: 1-500)
- **BB StdDev**: Standard deviation multiplier (default: 2.0, range: 0.1-5.0)

#### MACD Settings
- **MACD Fast**: Fast EMA period (default: 12, range: 1-100)
- **MACD Slow**: Slow EMA period (default: 26, range: 1-100)
- **MACD Signal**: Signal line period (default: 9, range: 1-100)

#### Stochastic Settings
- **Stoch %K**: K period (default: 14, range: 1-100)
- **Stoch %D**: D period (default: 3, range: 1-100)
- **Stoch Smooth**: Smoothing period (default: 3, range: 1-100)

#### ATR Settings
- **ATR Length**: Average true range period (default: 14, range: 1-100)

#### Support/Resistance Settings
- **S/R Lookback Bars**: Bars to analyze for pivots (default: 5, range: 2-50)
- **S/R Sensitivity**: Detection sensitivity (default: 0.3%, range: 0.1%-5.0%)

#### Display Settings
- **Show Base Indicators**: Toggle RSI, MACD, Stochastic, Bollinger Bands display
- **Show Custom Indicators**: Toggle custom composite indicators
- **Show Pending Order Levels**: Toggle buy/sell pending levels and stop losses
- **Show ML Data Table**: Toggle real-time data table display

### Step 3: Interpret the Indicator Output

#### ML Data Table (Top Right)

The table displays real-time values:

**Base Indicators Section:**
- RSI: Range 0-100 (>70 overbought, <30 oversold)
- Stoch %K: Range 0-100 (>80 overbought, <20 oversold)
- MACD: Difference between fast and slow EMAs
- BB Width: Distance between upper and lower Bollinger Bands

**Custom Indicators Section:**
- Momentum Score: Range -100 to +100 (positive = bullish, negative = bearish)
- Volatility Index: Range 0-100 (higher = more volatile)
- RSI Convergence: Range 0-100 (higher = more extreme RSI)
- Composite Signal: Weighted combination of all indicators

**Order Levels Section:**
- Buy Pending Level: Suggested buy entry point
- Sell Pending Level: Suggested sell entry point
- Buy Stop Loss: Stop loss below buy pending level
- Sell Stop Loss: Stop loss above sell pending level

#### Chart Visualization

**Price Chart (Overlay):**
- Yellow lines: Buy/Sell pending order levels
- Dashed lines: Take profit targets
- Current price relative to levels

**Indicator Panel:**
- Blue line: RSI
- Gray dotted lines: 30/70 reference levels
- Orange line: Bollinger Bands basis (EMA)
- Cyan lines: Upper/Lower Bollinger Bands
- Green area: Momentum Confluence (bullish)
- Red area: Volatility Expansion Index
- Purple area: RSI Convergence
- Yellow line: Composite Signal

## Part 2: Trading with the Indicator

### Identifying Trading Setups

#### Strong Buy Setup
Conditions:
- Composite Signal < -50
- RSI < 30
- Stochastic %K < 20
- Momentum Confluence negative and strong

Action:
- Place buy order at Buy Pending Level
- Set stop loss at Buy Stop Loss
- Target take profit at Recent High

#### Strong Sell Setup
Conditions:
- Composite Signal > +50
- RSI > 70
- Stochastic %K > 80
- Momentum Confluence positive and strong

Action:
- Place sell order at Sell Pending Level
- Set stop loss at Sell Stop Loss
- Target take profit at Recent Low

### Order Management

#### Buy Order Structure
```
Entry Price: Buy Pending Level
Quantity: Risk percentage based (e.g., 2-5% of account)
Stop Loss: Buy Stop Loss (ATR * 1.0 below pending level)
Take Profit: Sell Pending Level (1:2 risk-reward ratio)
```

#### Sell Order Structure
```
Entry Price: Sell Pending Level
Quantity: Risk percentage based (e.g., 2-5% of account)
Stop Loss: Sell Stop Loss (ATR * 1.0 above pending level)
Take Profit: Buy Pending Level (1:2 risk-reward ratio)
```

## Part 3: Data Export for ML Training

### Step 1: Collect Data

1. Run indicator on 15-minute to 4-hour timeframe
2. Collect data for minimum 3-6 months
3. Ensure sufficient trades occurred (target: 2000+ candles)
4. Record both filled and unfilled orders with outcomes

### Step 2: Export Data

Manual export from indicator table:

1. Open chart with indicator
2. Note values from ML Data Table each candle
3. Create CSV file with headers:

```
datetime,rsi,stoch,macd,bb_width,momentum_score,volatility_index,rsi_convergence,composite_signal,close_price,buy_pending_level,sell_pending_level,order_filled,order_profitable
```

4. Fill in values:

```
2024-12-31 09:00,45.23,62.15,0.15,0.85,12.45,78.32,45.67,25.14,1.0900,1.0850,1.0920,1,1
2024-12-31 09:15,46.12,64.31,0.18,0.87,15.23,79.15,48.92,28.45,1.0905,1.0840,1.0930,0,0
```

### Step 3: Label Historical Data

For each order generated by the indicator:

1. **order_filled**: 1 if order was triggered, 0 otherwise
2. **order_profitable**: 1 if trade closed with profit, 0 if loss or unfilled

Review historical trades and mark accurately for model training.

## Part 4: Python ML Training

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas scikit-learn numpy matplotlib seaborn
```

### Basic Training

```python
from ml_training_script import MLDataHandler, MLModelTrainer, OrderPredictor
from sklearn.model_selection import train_test_split

# Load data
handler = MLDataHandler()
handler.load_data('your_exported_data.csv')
handler.preprocess_data()
handler.feature_engineering()

# Prepare for ML
X, y, feature_cols = handler.prepare_ml_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
trainer = MLModelTrainer()
trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
trainer.train_pending_level_regressor(X_train, y_train, X_test, y_test)

# Evaluate
trainer.evaluate_all_models(X_test, y_test)

# Save models
trainer.save_models('./models')
```

### Making Predictions

```python
# Load trained models
predictor = OrderPredictor(trainer, feature_cols)

# Create feature dictionary from current indicator values
current_indicators = {
    'rsi': 45.23,
    'stoch': 62.15,
    'macd': 0.15,
    'bb_width': 0.85,
    'momentum_score': 12.45,
    'volatility_index': 78.32,
    'rsi_convergence': 45.67,
    'composite_signal': 25.14,
    'momentum_change': 0.5,
    'rsi_slope': 1.2,
    'volatility_ratio': 1.05,
    'price_to_buy_distance': 0.002,
    'price_to_sell_distance': 0.003,
    'order_fill_rate': 0.65,
    'order_profit_rate': 0.58
}

# Get prediction
prediction = predictor.predict_order_signal(current_indicators)

print(f"Fill Probability: {prediction['order_fill_probability']:.2%}")
print(f"Profit Probability: {prediction['order_profit_probability']:.2%}")
print(f"Recommendation: {prediction['recommendation']}")

# Recommendations
# STRONG_BUY: High confidence (fill > 70%, profit > 60%)
# BUY: Good setup (fill > 60%, profit > 50%)
# WATCH: Marginal setup (fill > 40%, profit > 50%)
# HOLD: Wait for better setup
```

## Recommended Timeframe Settings

### Scalping (1-5 minute bars)
```
RSI Length: 7
BB Length: 10
S/R Lookback: 3
ATR Length: 7
Entry trigger: Composite < -60 or > +60
```

### Day Trading (15-minute bars)
```
RSI Length: 9
BB Length: 12
S/R Lookback: 4
ATR Length: 9
Entry trigger: Composite < -50 or > +50
```

### Swing Trading (4-hour bars)
```
RSI Length: 14 (default)
BB Length: 20 (default)
S/R Lookback: 5 (default)
ATR Length: 14 (default)
Entry trigger: Composite < -50 or > +50
```

### Position Trading (Daily bars)
```
RSI Length: 21
BB Length: 30
S/R Lookback: 10
ATR Length: 20
Entry trigger: Composite < -40 or > +40
```

## Risk Management Rules

1. Risk only 2-5% per trade
2. Always use stop losses from indicator
3. Maintain 1:2 risk-reward ratio minimum
4. Skip trades during major news events
5. Reduce position size in high volatility
6. Close partial positions at 1:1 ratio
7. Trail stop loss using ATR as grid

## Model Maintenance

### Retraining Schedule

- Monthly: Quick model refresh
- Quarterly: Full retraining with new data
- After major market shifts: Immediate retraining

### Performance Monitoring

1. Track actual fill rates vs predicted
2. Track profit rates vs predicted
3. Monitor prediction accuracy
4. Adjust sensitivity if accuracy drops
5. Document market regime changes

## Troubleshooting

### Issue: Indicator not showing
- Check TradingView version (must support V5)
- Verify all inputs are within range
- Clear cache and reload page
- Check for compilation errors in script editor

### Issue: Low prediction accuracy
- Increase training data (minimum 2000 candles)
- Check for data quality issues (gaps, errors)
- Verify order_filled and order_profitable labels are correct
- Try different model parameters
- Consider adding new features

### Issue: Pending levels too far from price
- Reduce S/R Lookback bars
- Increase S/R Sensitivity
- Adjust ATR multiple (currently 0.5)
- Use tighter timeframe

## CSV Export Template

Create file with this exact format:

```csv
datetime,rsi,stoch,macd,bb_width,momentum_score,volatility_index,rsi_convergence,composite_signal,close_price,buy_pending_level,sell_pending_level,order_filled,order_profitable
2024-12-31 09:00:00,45.23,62.15,0.15,0.85,12.45,78.32,45.67,25.14,1.0900,1.0850,1.0920,1,1
2024-12-31 09:15:00,46.12,64.31,0.18,0.87,15.23,79.15,48.92,28.45,1.0905,1.0840,1.0930,0,0
2024-12-31 09:30:00,44.89,60.72,0.12,0.83,8.34,75.28,42.15,20.56,1.0895,1.0860,1.0910,1,0
```

## Advanced Configuration

### Custom Feature Engineering

Modify `ml_training_script.py` to add new features:

```python
def feature_engineering(self):
    # Add your custom features here
    self.data['custom_feature_1'] = # Your calculation
    self.data['custom_feature_2'] = # Your calculation
    # Then update prepare_ml_data() feature list
```

### Model Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.5]
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## Performance Benchmarks

Typical expected performance metrics:

- **Order Fill Prediction**: 60-75% accuracy
- **Profit Prediction**: 55-70% accuracy  
- **Level Prediction**: 0.5-1.5% mean absolute error
- **Win Rate**: 50-65% with proper risk management
- **Profit Factor**: 1.5-2.5 with good settings

## Support and Updates

Visit GitHub repository for:
- Latest code
- Bug reports
- Feature requests
- Community discussions

https://github.com/caizongxun/ML-Trading-Indicator
