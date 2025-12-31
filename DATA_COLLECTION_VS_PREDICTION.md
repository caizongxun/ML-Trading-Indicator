# 數據收集 vs 實時預測工作流 - 完整指南

## 核心概念澄清

你的問題非常關鍵：**數據收集和模型預測是兩個完全不同的階段**

---

## 階段1: 離線訓練 (Offline Training Phase)

### 1.1 數據收集

**目的**: 收集歷史K棒數據用於訓練模型

**方法**:
```
選擇時間段 (例如2024年全年)
    ↓
打開圖表 (EURUSD 15分鐘級別)
    ↓
添加Pine Script指標
    ↓
手動複製表格數據 (Data Window)
    ↓
粘貼到Excel → 保存為CSV
    ↓
運行Python清理腳本
    ↓
得到 training_data.csv (例如: 30,000行K棒)
```

**輸出範例 (training_data.csv)**:
```
DateTime,Open,High,Low,Close,Volume,RSI,StochK,StochD,...,LII,MTRI,DSRFI
2024-01-01 09:00:00,1.0850,1.0875,1.0840,1.0865,1250,55.2,48.3,51.5,...,45.2,62.1,38.9
2024-01-01 09:15:00,1.0865,1.0890,1.0860,1.0880,980,58.7,52.1,54.8,...,42.8,65.3,41.2
2024-01-01 09:30:00,1.0880,1.0885,1.0870,1.0875,1150,57.3,50.9,53.2,...,44.1,63.7,39.8
...
(共30,000行)
```

### 1.2 模型訓練

**ML模型學習什麼**:
```
輸入特徵 (當前K棒的技術指標):
├─ RSI = 57.3
├─ StochK = 50.9
├─ MACD = 0.0025
├─ BB Upper/Lower = 1.0920/1.0810
├─ ATR = 0.0035
└─ ... 15個其他特徵

模型學習:
「在這些指標組合下,掛單有多大機率會被觸發?」
「在這些指標組合下,掛單有多大機率會盈利?」
「最優的買入/賣出掛單點位應該在哪裡?」

輸出預測:
├─ order_fill_probability = 0.72 (72% 機率被觸發)
├─ order_profit_probability = 0.68 (68% 機率盈利)
├─ buy_pending_level = 1.0840
└─ sell_pending_level = 1.0890
```

**訓練過程**:
```python
# 步驟1: 加載歷史數據
handler = MLDataHandler('training_data.csv')
handler.load_data()
handler.preprocess_data()
handler.feature_engineering()

# 步驟2: 準備特徵和標籤
X, y, feature_cols = handler.prepare_ml_data()
# X = (30000, 15) - 30,000個K棒的15個特徵
# y = (30000, 2) - 對應的標籤 (是否被觸發, 是否盈利)

# 步驟3: 訓練模型
trainer = MLModelTrainer()
trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
trainer.train_pending_level_regressor(X_train, y_train, X_test, y_test)

# 步驟4: 保存模型
trainer.save_models()  # → order_filled_model.pkl, order_profitable_model.pkl等
```

---

## 階段2: 實時預測 (Real-Time Prediction Phase)

### 2.1 實時操作

**目的**: 使用訓練好的模型預測新的K棒何時應該開單

**工作流程**:
```
實時監控圖表 (每15分鐘一根新K棒)
    ↓
Pine Script指標自動計算最新K棒的24個指標值
    ↓
你看到Data Window中的新數據
    ↓
運行Python預測腳本:
  - 讀取當前K棒的15個特徵
  - 傳給已訓練的模型
  - 模型輸出預測
    ↓
獲得實時建議:
  ✓ "STRONG_BUY" - 在1.0840開買單
  ✓ "BUY" - 考慮開多
  ✓ "WATCH" - 觀望
  ✓ "HOLD" - 不開單
```

### 2.2 實時預測代碼

```python
import json
from datetime import datetime
from ML_Trading_Data_Exporter import MLModelTrainer, OrderPredictor

# 步驟1: 加載已訓練的模型
trainer = MLModelTrainer()
trainer.load_models('./models')  # 加載保存的模型

# 定義特徵列
feature_cols = [
    'rsi', 'stoch', 'macd', 'bb_width',
    'momentum_score', 'volatility_index', 'rsi_convergence',
    'composite_signal', 'momentum_change', 'rsi_slope',
    'volatility_ratio', 'price_to_buy_distance',
    'price_to_sell_distance', 'order_fill_rate', 'order_profit_rate'
]

# 步驟2: 初始化預測器
predictor = OrderPredictor(trainer, feature_cols)

# 步驟3: 定期檢查新K棒
# (這可以通過定時任務或webhook實現)

def check_latest_bar():
    # 從TradingView Data Window複製的最新數據
    current_bar_data = {
        'datetime': datetime.now(),
        'rsi': 57.3,
        'stoch': 50.9,
        'macd': 0.0025,
        'bb_width': 0.0110,
        'momentum_score': 5.2,
        'volatility_index': 42.1,
        'rsi_convergence': 6.4,
        'composite_signal': 2.8,
        'momentum_change': 3.1,
        'rsi_slope': 2.5,
        'volatility_ratio': 1.05,
        'price_to_buy_distance': 0.002,
        'price_to_sell_distance': 0.003,
        'order_fill_rate': 0.62,
        'order_profit_rate': 0.58
    }
    
    # 步驟4: 使用模型預測
    prediction = predictor.predict_order_signal(current_bar_data)
    
    # 步驟5: 輸出結果
    print(f"時間: {prediction['timestamp']}")
    print(f"開單機率: {prediction['order_fill_probability']:.2%}")
    print(f"盈利機率: {prediction['order_profit_probability']:.2%}")
    print(f"推薦: {prediction['recommendation']}")
    
    if prediction['recommendation'] in ['STRONG_BUY', 'BUY']:
        print(f"買入點位: {prediction['buy_pending_level']:.5f}")
        print(f"賣出點位: {prediction['sell_pending_level']:.5f}")
    
    return prediction

# 每次新K棒結束時調用
if __name__ == "__main__":
    result = check_latest_bar()
    print(json.dumps(result, indent=2))
```

---

## 實際場景對比

### 場景1: 數據收集 (第一次)

```
週一-週五 (整個工作週)
    ↓
手動複製2024年全年的EURUSD 15分鐘K棒數據
    ↓
大約30,000根K棒
    ↓
保存為 historical_data_2024.csv
    ↓
訓練模型 (耗時5-10分鐘)
    ↓
模型保存到磁盤
```

### 場景2: 實時預測 (每天)

```
週一上午 09:00
    ↓
Pine Script自動計算9:00的15分鐘K棒
    ↓
Data Window顯示該K棒的24個指標
    ↓
你複製這一行數據
    ↓
運行Python腳本 (1秒內)
    ↓
立即獲得預測結果:
  "STRONG_BUY at 1.0840"
    ↓
在MetaTrader/Interactive Brokers中下單
    ↓
09:15 新K棒產生
    ↓
重複上述過程
```

---

## 數據流程圖

```
┌─────────────────────────────────────────────────────────────┐
│              階段1: 離線訓練 (一次性)                        │
└─────────────────────────────────────────────────────────────┘

TradingView圖表 (2024年全年)
    ↓
Pine Script指標
    ↓ (複製表格)
Data Window (30,000行K棒)
    ↓
Excel/Google Sheets
    ↓
CSV: training_data.csv
    ↓
Python: MLDataHandler
    ↓
ML Models (Random Forest, Gradient Boosting)
    ↓
Saved Models (./models/)
    ↓ (產生一次,永久保存)


┌─────────────────────────────────────────────────────────────┐
│           階段2: 實時預測 (每天多次)                        │
└─────────────────────────────────────────────────────────────┘

TradingView實時圖表 (每15分鐘)
    ↓
Pine Script計算最新K棒指標
    ↓ (複製1行)
Data Window (當前K棒1行)
    ↓
Python OrderPredictor
    ↓
已訓練的模型 (from ./models/)
    ↓
實時預測結果 (1秒)
    ↓
動作: 開買單 / 開賣單 / 持觀望
```

---

## 常見問題解答

### Q1: 我需要每根K棒都複製嗎?

**A**: 
- **訓練階段**: 是的,需要複製所有歷史K棒 (但只需做一次)
- **預測階段**: 不是,只需複製最新的1根K棒來獲取預測

### Q2: Data Window複製的就是我需要的數據嗎?

**A**: 是的! Data Window中的數據格式完全符合CSV標準:
```
DateTime,Open,High,Low,Close,Volume,RSI,StochK,...
2024-01-01 09:00,1.0850,1.0875,1.0840,1.0865,1250,55.2,48.3,...
2024-01-01 09:15,1.0865,1.0890,1.0860,1.0880,980,58.7,52.1,...
```

直接複製粘貼到Excel即可。

### Q3: 模型訓練後,我還需要複製數據嗎?

**A**: 
- **如果只是預測**: 只需複製最新1根K棒的1行數據
- **如果要重新訓練**: 需要複製新的歷史數據(例如新增3個月的數據)

### Q4: 能自動化嗎?不手動複製?

**A**: 可以的! 有三種方案:

**方案1: TradingView API (付費)**
```python
# 使用官方API自動拉取數據
import requests
history = requests.get('https://www.tradingview.com/api/...', params=...)
```

**方案2: 第三方數據源 (免費/付費)**
```python
import yfinance as yf
# 自動下載EURUSD數據 (需要轉換符號)
data = yf.download('EURUSD=X', start='2024-01-01', end='2024-12-31')
```

**方案3: MetaTrader 4/5 API**
```mql5
// 在MT4/MT5中運行,直接輸出CSV
// 然後用Python讀取該CSV
```

### Q5: 模型準確度如何評估?

**A**: 在訓練數據上評估:
```python
# 在測試集上評估 (模型沒看過這些數據)
accuracy = model.score(X_test, y_test)
print(f"模型準確率: {accuracy:.2%}")  # 例如: 73.45%

# ROC-AUC評分 (0.5=隨機, 1.0=完美)
roc_auc = roc_auc_score(y_test, pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")  # 例如: 0.7823
```

---

## 推薦工作流程

### 第1週: 準備階段
```
1. 收集歷史數據 (2-3天)
   - 選擇交易對 (EURUSD)
   - 選擇時間級別 (15分鐘)
   - 選擇時間段 (至少6個月,最好1年)
   - 複製所有K棒 → CSV

2. 數據清理和訓練 (1-2小時)
   - 運行 tick_data_processor.py
   - 運行 ML_Trading_Data_Exporter.py
   - 模型自動保存到 ./models/

3. 驗證模型 (30分鐘)
   - 查看準確率
   - 查看ROC-AUC分數
```

### 第2週+: 交易階段
```
每天交易時間:
1. 09:00 - 等待第一根K棒完成
2. 09:15 - 複製該K棒的1行數據
3. 09:15 - 運行預測腳本 (1秒)
4. 09:16 - 根據預測下單
5. 09:30 - 等待第二根K棒...
```

---

## 實時預測腳本範本 (實用)

保存為 `real_time_prediction.py`:

```python
import pandas as pd
import json
from datetime import datetime
from ML_Trading_Data_Exporter import MLModelTrainer, OrderPredictor

class RealTimePredictionEngine:
    def __init__(self):
        self.trainer = MLModelTrainer()
        self.trainer.load_models('./models')
        self.feature_cols = [
            'rsi', 'stoch', 'macd', 'bb_width',
            'momentum_score', 'volatility_index', 'rsi_convergence',
            'composite_signal', 'momentum_change', 'rsi_slope',
            'volatility_ratio', 'price_to_buy_distance',
            'price_to_sell_distance', 'order_fill_rate', 'order_profit_rate'
        ]
        self.predictor = OrderPredictor(self.trainer, self.feature_cols)
    
    def predict_from_csv_line(self, csv_line):
        """
        從複製的Data Window單行進行預測
        
        使用方式:
        1. 在TradingView複製一行數據
        2. engine.predict_from_csv_line("粘貼的內容")
        """
        # 解析CSV行
        values = csv_line.strip().split(',')
        
        current_bar = {
            'rsi': float(values[6]),
            'stoch': float(values[7]),
            'macd': float(values[9]),
            'bb_width': float(values[12]) - float(values[14]),
            'momentum_score': float(values[16]),
            'volatility_index': float(values[17]),
            'rsi_convergence': float(values[18]),
            'composite_signal': float(values[19]),
            'momentum_change': 0.0,  # 需要歷史數據計算
            'rsi_slope': 0.0,        # 需要歷史數據計算
            'volatility_ratio': float(values[17]) / 40.0,  # 簡化
            'price_to_buy_distance': 0.002,  # 簡化
            'price_to_sell_distance': 0.003,  # 簡化
            'order_fill_rate': 0.6,   # 需要滾動計算
            'order_profit_rate': 0.55  # 需要滾動計算
        }
        
        return self.predictor.predict_order_signal(current_bar)

# 使用示例
if __name__ == "__main__":
    engine = RealTimePredictionEngine()
    
    # 從TradingView複製的Data Window行
    data_line = "2024-12-31 14:30,1.0850,1.0875,1.0840,1.0865,1250,57.3,50.9,53.2,0.0025,0.0020,0.0005,1.0920,1.0865,1.0810,0.0035,5.2,42.1,6.4,2.8,45.2,62.1,38.9"
    
    result = engine.predict_from_csv_line(data_line)
    
    print("="*60)
    print(f"預測時間: {result['timestamp']}")
    print(f"開單機率: {result['order_fill_probability']:.2%}")
    print(f"盈利機率: {result['order_profit_probability']:.2%}")
    print(f"推薦: {result['recommendation']}")
    print("="*60)
    
    if result['recommendation'] in ['STRONG_BUY', 'BUY']:
        print(f"\n✓ 建議開單!")
        print(f"  買入點位: {result['buy_pending_level']:.5f}")
        print(f"  賣出點位: {result['sell_pending_level']:.5f}")
        print(f"  潛在盈利: {(result['sell_pending_level'] - result['buy_pending_level'])*100:.2f} pips")
    else:
        print(f"\n✗ 暫不建議開單 (等待更好機會)")
```

---

## 總結

| 階段 | 操作 | 頻率 | 時間 | 數據量 |
|------|------|------|------|--------|
| **訓練** | 收集歷史K棒 + 訓練模型 | 一次 | 2-3小時 | 30,000行 |
| **預測** | 複製新K棒 + 運行預測 | 每15分鐘 | 1秒 | 1行 |

關鍵點:
1. ✓ 數據收集只需做一次 (用於訓練)
2. ✓ 實時預測每次只需1行新數據
3. ✓ 模型學會了指標組合與掛單點位的關係
4. ✓ 可以自動化(進階方案)
