# ML模型训练与新指标集成手册

## 步骤1: 从TradingView导出数据方法

### 1.1 添加Pine Script指标到你的图表

```
1. 打开TradingView
2. 第1步: 创建一个新的Pine Script
3. 不会创建？仄上选择 "New indicator" (Pine Script v5)
4. 将此项目中的 ML_Order_Helper_V5.pine 的整个代码复制进去
```

### 1.2 添加指标到你的图表

```
1. 添加指标不是更新指标 (Update indicator) - 需要保存为Script或Private
2. 国三/国三最住位 (EURUSD, GBPUSD, AUDUSD)图表添加
3. 伏级住上设置: Data Window = On
```

### 1.3 导出国表数据并保存为CSV

```
1. 滺轮减疮下方的表格区域
2. 然后全选择 (Ctrl+A)
3. 复制整个表格并保存为: trading_data.csv
```

## 步骤2: Python环境配置

### 2.1 安装必需的包

```bash
# 安装scikit-learn和数据兀7个

pip install pandas numpy scikit-learn matplotlib seaborn

# 可选: 使用加速窗口
# pip install -U scikit-learn  # 更新到最新版本
```

### 2.2 塊共了整个ML训练脚本

```bash
# 下载了整个项目后

cd ML-Trading-Indicator

# 你知林深幔せているフォルダ结构
```

## 步骤3: 更新类不配置 (feature_engineering)

### 3.1 修改 `ML_Trading_Data_Exporter.py`

低找到这个方法：

```python
def feature_engineering(self):
    """Feature Engineering: Create new features"""
    # 原有代码...
```

下面在此方法的 `print("Feature Engineering completed")` 前面添加：

```python
def feature_engineering(self):
    """Feature Engineering: Create new features"""
    
    # 原有特征核心... (保持原样)
    self.data['momentum_change'] = self.data['momentum_score'].diff().fillna(0)
    self.data['rsi_slope'] = self.data['rsi'].diff().fillna(0)
    self.data['volatility_ratio'] = (
        self.data['bb_width'] / self.data['bb_width'].rolling(20).mean()
    ).fillna(1)
    self.data['price_to_buy_distance'] = (
        (self.data['buy_pending_level'] - self.data['close_price']) / 
        self.data['close_price']
    )
    self.data['price_to_sell_distance'] = (
        (self.data['sell_pending_level'] - self.data['close_price']) / 
        self.data['close_price']
    )
    self.data['order_fill_rate'] = (
        self.data['order_filled'].rolling(50).mean()
    ).fillna(0)
    self.data['order_profit_rate'] = (
        self.data['order_profitable'].rolling(50).mean()
    ).fillna(0)
    
    # ===== NEW: Add 3 Advanced ML Indicators =====
    from advanced_indicators import AdvancedIndicators
    
    indicator_calc = AdvancedIndicators(self.data)
    self.data['lii'] = indicator_calc.calculate_lii()
    self.data['mtri'] = indicator_calc.calculate_mtri()
    self.data['dsrfi'] = indicator_calc.calculate_dsrfi()
    
    print("✓ Advanced ML indicators added: LII, MTRI, DSRFI")
    print("="*60)
    
    return self.data
```

### 3.2 修改 `prepare_ml_data()` 方法

低找到：

```python
def prepare_ml_data(self):
    """Prepare ML training features and labels"""
    feature_cols = [
        'rsi', 'stoch', 'macd', 'bb_width', 
        'momentum_score', 'volatility_index', 'rsi_convergence', 
        'composite_signal', 'momentum_change', 'rsi_slope',
        'volatility_ratio', 'price_to_buy_distance', 
        'price_to_sell_distance', 'order_fill_rate', 'order_profit_rate'
    ]
```

修改为：

```python
def prepare_ml_data(self):
    """Prepare ML training features and labels"""
    feature_cols = [
        # Original indicators
        'rsi', 'stoch', 'macd', 'bb_width', 
        'momentum_score', 'volatility_index', 'rsi_convergence', 
        'composite_signal', 'momentum_change', 'rsi_slope',
        'volatility_ratio', 'price_to_buy_distance', 
        'price_to_sell_distance', 'order_fill_rate', 'order_profit_rate',
        
        # NEW: 3 Advanced ML Indicators
        'lii',    # Liquidity Imbalance Index
        'mtri',   # Multi-Timeframe Resonance Index
        'dsrfi'   # Dynamic S/R Fractal Index
    ]
```

## 步骤4: 执行训练

### 4.1 准备数据文件

```bash
# 1. 将从TradingView导出的CSV文件保存为 trading_data.csv
# 2. 放在项目根目录下
```

### 4.2 修改主程序中main()

低找到：

```python
if __name__ == "__main__":
    trainer, handler, predictor = main()
```

修改而 handler.create_sample_data() 到 handler.load_data():

```python
if __name__ == "__main__":
    # Step 1: Data Loading and Preparation
    print("Step 1: Data Loading and Preparation")
    print("-" * 60)
    
    handler = MLDataHandler()
    # Load actual data from TradingView
    if handler.load_data('trading_data.csv'):
        handler.preprocess_data()
        handler.feature_engineering()
    else:
        print("Warning: Could not load data, using sample data")
        handler.create_sample_data(n_samples=1500)
        handler.preprocess_data()
        handler.feature_engineering()
    
    # ... rest of the code remains the same
    trainer, handler, predictor = main()
```

### 4.3 执行训练

```bash
python ML_Trading_Data_Exporter.py
```

输出示例：

```
============================================================
ML Training Data Processing and Model Training System
============================================================

Step 1: Data Loading and Preparation
------------------------------------------------------------
Loading data from trading_data.csv...
✓ Successfully loaded 1000 K-line data
  Columns: [..., 'lii', 'mtri', 'dsrfi']
✓ Data cleaning completed, 995 rows retained
✓ Advanced ML indicators added: LII, MTRI, DSRFI
============================================================

Step 2: Prepare ML Data
------------------------------------------------------------
✓ Preparation completed: 995 samples, 18 features
  

Step 3: Model Training
------------------------------------------------------------

============================================================
Train Model 1: Pending Order Fill Probability (Classification)
============================================================

  Training Logistic Regression...
    Train Accuracy: 0.7234
    Test Accuracy: 0.7156
    ROC-AUC: 0.8012
    
  ...
  
  Feature Importance (Top 10):
    1. dsrfi          0.0854
    2. mtri           0.0742
    3. lii            0.0691
    4. ...
```

## 步骤5: 验识特征重要性

训练完成后，检查新指标是否提高了模型性能：

```python
# 在main()脚本末尾添加

print("\n" + "="*60)
print("特征重要性分析")
print("="*60)

results = trainer.results['order_filled']
if results['feature_importance'] is not None:
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': results['feature_importance']
    }).sort_values('importance', ascending=False)
    
    print("\n最重要的初10个特征:")
    print(importance_df.head(10).to_string())
    
    # 检查新指标是否在前5
    new_indicators = ['lii', 'mtri', 'dsrfi']
    for ind in new_indicators:
        rank = importance_df[importance_df['feature'] == ind].index[0] + 1 if ind in importance_df['feature'].values else len(importance_df)
        print(f"  {ind.upper()}: Rank #{rank}")
```

## 步骤6: 使用训练好的模型预测

### 6.1 实时预测示例

```python
# 触发条件（例如每1小时）
current_candle_features = {
    'rsi': 25.5,
    'stoch': 18.2,
    'macd': -0.0023,
    'bb_width': 1.8,
    'lii': 78.5,      # NEW
    'mtri': 72.3,     # NEW
    'dsrfi': 81.2,    # NEW
    # ... other features
}

prediction = predictor.predict_order_signal(current_candle_features)

print(f"Order Fill Probability: {prediction['order_fill_probability']:.2%}")
print(f"Order Profit Probability: {prediction['order_profit_probability']:.2%}")
print(f"Buy Level: {prediction['buy_pending_level']:.5f}")
print(f"Sell Level: {prediction['sell_pending_level']:.5f}")
print(f"Recommendation: {prediction['recommendation']}")
```

## 步骤7: 见片上应用 (Pine Script)

你子可以在Pine Script中添加预测逻辑来像控制位捇算法一样使用ML的结果（例子：

```pinescript
// 基于ML模型的预测 - 值输出为0-1
ml_fill_prob = close > close[1] ? 0.75 : 0.45  // Placeholder for API
ml_profit_prob = close > close[1] ? 0.68 : 0.32

// 预测掛单点位
(lii > 70 and mtri > 60 and dsrfi > 70 and ml_fill_prob > 0.7) ? 
    strategy.entry("Buy", strategy.long) : na
```

## 步骤8: 分析结果

训练完成后，你氢拥有:

1. **三个模型**
   - `order_filled_model.pkl` - 掛单填充概率
   - `order_profitable_model.pkl` - 掛单盈利概率
   - `buy_pending_level_model.pkl` - 买入点位
   - `sell_pending_level_model.pkl` - 卖出点位

2. **结果文件**
   - `models/training_results.json` - 训练结果氢壊

3. **性能指标**
   - 模型準确率
   - ROC-AUC分数
   - 特征重要性

## ⚠️ 常见问题

**Q: 总是报错 "lii not found"**
A: 确保你已经从TradingView导出了包含lii, mtri, dsrfi列的数据，或从Python第一次不会自动计算。

**Q: 模型準确率不高总是55%**
A: 
1. 检查数据量 - 最似要至尐1000个样本
2. 检查数据质量 - 是否有正常的成交量和价格步伏
3. 收集更井提供不同个自事件时斵的数据

**Q: 有没有例子数据集？**
A: 没有。你需要花时间从TradingView导出真实交易数据。
