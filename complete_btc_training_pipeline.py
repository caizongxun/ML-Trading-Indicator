"""
BTC 完整訓練管道
用途：從 HuggingFace 下載BTC 15分鐘K棒數據 → 計算所有指標 → 訓練ML模型 → 預測最優開單點位
版本：2.0
要求：Python 3.8+, pandas, scikit-learn, matplotlib, pyarrow, huggingface_hub

使用方式：
    python complete_btc_training_pipeline.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

# 數據下載
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("請安裝: pip install huggingface_hub")

# 機器學習相關
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# SECTION 0: 數據下載 (從HuggingFace)
# ============================================================================

class HuggingFaceDataLoader:
    """從HuggingFace下載BTC K棒數據"""
    
    def __init__(self):
        self.repo_id = "zongowo111/v2-crypto-ohlcv-data"
        self.file_path = "klines/BTCUSDT/BTC_15m.parquet"
        self.data = None
    
    def download_and_load(self, local_cache_dir='./data'):
        """
        從HuggingFace下載並加載BTC數據
        
        Returns:
            pd.DataFrame: BTC K棒數據
        """
        print(f"\n正在從HuggingFace下載 {self.repo_id}...")
        
        try:
            # 下載文件
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.file_path,
                cache_dir=local_cache_dir,
                repo_type="dataset"
            )
            
            print(f"✓ 下載完成: {file_path}")
            
            # 加載Parquet文件
            self.data = pd.read_parquet(file_path)
            
            print(f"✓ 數據加載成功")
            print(f"  數據行數: {len(self.data)}")
            print(f"  時間範圍: {self.data.index.min()} 到 {self.data.index.max()}")
            print(f"  列: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            print(f"✗ 下載失敗: {e}")
            print(f"  請確保已安裝 pyarrow: pip install pyarrow")
            return None
    
    def prepare_dataframe(self):
        """
        準備DataFrame - 統一列名並轉換時間格式
        """
        if self.data is None:
            return None
        
        df = self.data.copy()
        
        # 統一列名 (小寫)
        df.columns = df.columns.str.lower()
        
        # 如果index是時間戳，轉換為datetime
        if df.index.name == 'open_time' or not isinstance(df.index, pd.DatetimeIndex):
            if isinstance(df.index, pd.RangeIndex):
                # 如果有timestamp列，使用它
                if 'timestamp' in df.columns:
                    df.index = pd.to_datetime(df['timestamp'], unit='ms')
                elif 'open_time' in df.columns:
                    df.index = pd.to_datetime(df['open_time'], unit='ms')
                else:
                    # 嘗試從列名推斷
                    for col in df.columns:
                        if 'time' in col.lower():
                            df.index = pd.to_datetime(df[col], unit='ms')
                            break
        
        # 確保列名統一
        rename_dict = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'volume': 'volume',
            'quote_asset_volume': 'quote_volume',
            'number_of_trades': 'trades'
        }
        
        df = df.rename(columns=rename_dict)
        
        # 保留必要的OHLCV列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]
        
        self.data = df
        return df

# ============================================================================
# SECTION 1: 指標計算
# ============================================================================

class IndicatorCalculator:
    """計算所有技術指標"""
    
    def __init__(self, data):
        self.data = data.copy()
    
    # ====== 基礎指標 ======
    
    def calculate_rsi(self, period=14):
        """RSI (相對強弱指數)"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, period=14):
        """Stochastic Oscillator"""
        low_min = self.data['low'].rolling(window=period).min()
        high_max = self.data['high'].rolling(window=period).max()
        
        k_percent = 100 * (self.data['close'] - low_min) / (high_max - low_min + 0.0001)
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """MACD (移動平均收斂發散)"""
        ema_fast = self.data['close'].ewm(span=fast).mean()
        ema_slow = self.data['close'].ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        macd_hist = macd_line - signal_line
        
        return macd_line, signal_line, macd_hist
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    def calculate_atr(self, period=14):
        """Average True Range (平均真實波幅)"""
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    # ====== 自創指標 ======
    
    def calculate_momentum_score(self, period=20):
        """動量分數 (-100 到 +100)"""
        # RSI貢獻度
        rsi = self.calculate_rsi(14)
        rsi_score = (rsi - 50) / 50  # 歸一化到 -1 到 1
        
        # 價格變化率
        returns = self.data['close'].pct_change(period)
        returns_score = np.clip(returns * 100, -1, 1)  # 歸一化
        
        # 合併
        momentum = ((rsi_score + returns_score) / 2) * 100
        return momentum.fillna(0)
    
    def calculate_volatility_index(self, period=20):
        """波動性指數 (0-100)"""
        atr = self.calculate_atr(14)
        atr_sma = atr.rolling(window=period).mean()
        
        # 波動率 = (當前ATR / 平均ATR - 1) * 100
        volatility = ((atr / (atr_sma + 0.0001)) - 1) * 100
        volatility = np.clip(volatility, -100, 100)
        
        # 轉換到 0-100 範圍
        volatility = (volatility + 100) / 2
        return volatility.fillna(50)
    
    def calculate_rsi_convergence(self):
        """RSI與Stochastic的收斂程度 (0-100)"""
        rsi = self.calculate_rsi(14)
        stoch_k, _ = self.calculate_stochastic(14)
        
        # 計算差異 (越小越收斂)
        difference = np.abs(rsi - stoch_k) / 100 * 100
        convergence = 100 - difference
        
        return convergence.fillna(50)
    
    def calculate_composite_signal(self):
        """綜合信號 (結合多個指標)"""
        rsi = self.calculate_rsi(14)
        stoch_k, _ = self.calculate_stochastic(14)
        macd_line, _, _ = self.calculate_macd()
        
        # RSI 信號 (-1 到 1)
        rsi_signal = (rsi - 50) / 50
        
        # Stochastic 信號 (-1 到 1)
        stoch_signal = (stoch_k - 50) / 50
        
        # MACD 信號 (歸一化)
        macd_signal = np.clip(macd_line / 0.05, -1, 1)  # 假設MACD範圍
        
        # 加權平均
        composite = (rsi_signal * 0.4 + stoch_signal * 0.3 + macd_signal * 0.3) * 100
        
        return composite.fillna(0)
    
    # ====== 高級ML指標 ======
    
    def calculate_lii(self, period=14):
        """流動性失衡指數 (0-100)"""
        buy_pressure = (self.data['close'] - self.data['low']) / (self.data['high'] - self.data['low'] + 0.0001)
        sell_pressure = (self.data['high'] - self.data['close']) / (self.data['high'] - self.data['low'] + 0.0001)
        
        imbalance = np.abs(buy_pressure - sell_pressure) * 100
        lii = imbalance.rolling(window=period).mean()
        
        return lii.fillna(50)
    
    def calculate_mtri(self, period=14):
        """多時間框架共鳴指數 (0-100)"""
        rsi14 = self.calculate_rsi(14)
        rsi7 = self.calculate_rsi(7)
        rsi21 = self.calculate_rsi(21)
        
        # 計算RSI協調強度
        correlation = (np.abs(rsi14 - 50) + np.abs(rsi7 - 50) + np.abs(rsi21 - 50)) / 3
        mtri = correlation
        
        return mtri.fillna(50)
    
    def calculate_dsrfi(self, period=14):
        """動態支撐阻力破裂指數 (0-100)"""
        atr = self.calculate_atr(14)
        volatility = (atr / self.data['close']) * 100
        
        # 計算趨勢強度 (上升K棒數 - 下降K棒數)
        up_count = (self.data['close'] > self.data['close'].shift(1)).rolling(window=period).sum()
        down_count = (self.data['close'] < self.data['close'].shift(1)).rolling(window=period).sum()
        
        trend_strength = np.abs((up_count - down_count) / period * 100)
        dsrfi = (trend_strength + volatility) / 2
        
        return dsrfi.fillna(50)
    
    def calculate_all(self):
        """
        計算所有指標
        """
        print("\n計算所有指標...")
        
        # 基礎指標
        self.data['rsi'] = self.calculate_rsi(14)
        stoch_k, stoch_d = self.calculate_stochastic(14)
        self.data['stoch_k'] = stoch_k
        self.data['stoch_d'] = stoch_d
        macd_line, signal_line, macd_hist = self.calculate_macd()
        self.data['macd'] = macd_line
        self.data['macd_signal'] = signal_line
        self.data['macd_hist'] = macd_hist
        
        bb_upper, bb_basis, bb_lower = self.calculate_bollinger_bands(20, 2)
        self.data['bb_upper'] = bb_upper
        self.data['bb_basis'] = bb_basis
        self.data['bb_lower'] = bb_lower
        self.data['bb_width'] = bb_upper - bb_lower
        
        self.data['atr'] = self.calculate_atr(14)
        
        # 自創指標
        self.data['momentum_score'] = self.calculate_momentum_score(20)
        self.data['volatility_index'] = self.calculate_volatility_index(20)
        self.data['rsi_convergence'] = self.calculate_rsi_convergence()
        self.data['composite_signal'] = self.calculate_composite_signal()
        
        # 高級ML指標
        self.data['lii'] = self.calculate_lii(14)
        self.data['mtri'] = self.calculate_mtri(14)
        self.data['dsrfi'] = self.calculate_dsrfi(14)
        
        # 移除NaN
        self.data = self.data.dropna()
        
        print(f"✓ 完成計算所有指標 ({len(self.data)} 有效行)")
        
        return self.data

# ============================================================================
# SECTION 2: 特徵工程 & 標籤生成
# ============================================================================

class FeatureEngineer:
    """特徵工程和標籤生成"""
    
    def __init__(self, data):
        self.data = data.copy()
    
    def generate_labels(self, lookahead=5, profit_threshold=0.001):
        """
        生成標籤
        
        Args:
            lookahead: 向前看多少根K棒
            profit_threshold: 盈利閾值 (0.001 = 0.1%)
        """
        print(f"\n生成標籤 (lookahead={lookahead}, profit_threshold={profit_threshold})...")
        
        # 計算未來最高價和最低價
        future_high = self.data['high'].rolling(window=lookahead).max().shift(-lookahead)
        future_low = self.data['low'].rolling(window=lookahead).min().shift(-lookahead)
        
        current_close = self.data['close']
        
        # 掛單點位設置
        buy_pending = current_close * (1 - 0.002)  # 買單掛在2個pips下方
        sell_pending = current_close * (1 + 0.002)  # 賣單掛在2個pips上方
        
        # 標籤1: 掛單是否被觸發
        order_filled = ((future_high >= buy_pending) | (future_low <= sell_pending)).astype(int)
        
        # 標籤2: 掛單是否盈利
        # 買單盈利: 觸發後價格上升超過閾值
        buy_filled = future_high >= buy_pending
        buy_profitable = (future_high >= (buy_pending * (1 + profit_threshold))).astype(int) & buy_filled.astype(int)
        
        # 賣單盈利: 觸發後價格下降超過閾值
        sell_filled = future_low <= sell_pending
        sell_profitable = (future_low <= (sell_pending * (1 - profit_threshold))).astype(int) & sell_filled.astype(int)
        
        # 合併
        order_profitable = (buy_profitable | sell_profitable).astype(int)
        
        self.data['order_filled'] = order_filled
        self.data['order_profitable'] = order_profitable
        self.data['buy_pending_level'] = buy_pending
        self.data['sell_pending_level'] = sell_pending
        
        # 移除標籤無效的行
        self.data = self.data.dropna()
        
        print(f"✓ 標籤生成完成")
        print(f"  被觸發的掛單: {order_filled.sum()} ({order_filled.sum()/len(order_filled)*100:.1f}%)")
        print(f"  盈利的掛單: {order_profitable.sum()} ({order_profitable.sum()/len(order_filled)*100:.1f}%)")
        
        return self.data
    
    def create_features(self):
        """
        創建機器學習特徵
        """
        print("\n創建ML特徵...")
        
        # 動態特徵
        self.data['momentum_change'] = self.data['momentum_score'].diff().fillna(0)
        self.data['rsi_slope'] = self.data['rsi'].diff().fillna(0)
        self.data['volatility_ratio'] = (
            self.data['bb_width'] / self.data['bb_width'].rolling(20).mean()
        ).fillna(1)
        
        # 價格相對位置
        self.data['price_to_buy_distance'] = (
            (self.data['buy_pending_level'] - self.data['close']) / self.data['close']
        )
        self.data['price_to_sell_distance'] = (
            (self.data['sell_pending_level'] - self.data['close']) / self.data['close']
        )
        
        # 滾動成功率
        self.data['order_fill_rate'] = (
            self.data['order_filled'].rolling(50).mean()
        ).fillna(0)
        self.data['order_profit_rate'] = (
            self.data['order_profitable'].rolling(50).mean()
        ).fillna(0)
        
        self.data = self.data.dropna()
        
        print(f"✓ 特徵創建完成 ({len(self.data)} 行)")
        return self.data
    
    def get_features_and_labels(self):
        """
        準備ML所需的特徵和標籤
        """
        feature_cols = [
            'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'atr', 'momentum_score', 'volatility_index',
            'rsi_convergence', 'composite_signal', 'lii', 'mtri', 'dsrfi',
            'momentum_change', 'rsi_slope', 'volatility_ratio',
            'price_to_buy_distance', 'price_to_sell_distance',
            'order_fill_rate', 'order_profit_rate'
        ]
        
        X = self.data[feature_cols].copy()
        y = self.data[['order_filled', 'order_profitable']].copy()
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        print(f"✓ 特徵準備完成: {X_scaled.shape[0]} 樣本, {X_scaled.shape[1]} 特徵")
        
        return X_scaled, y, feature_cols, scaler

# ============================================================================
# SECTION 3: 模型訓練
# ============================================================================

class MLModelTrainer:
    """機器學習模型訓練"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_order_filled_classifier(self, X_train, y_train, X_test, y_test):
        """
        訓練掛單填充分類器
        """
        print("\n" + "="*70)
        print("訓練模型 1: 掛單是否被觸發 (分類)")
        print("="*70)
        
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=self.random_state)
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, clf in classifiers.items():
            print(f"\n  訓練 {name}...")
            
            clf.fit(X_train, y_train['order_filled'])
            
            train_score = clf.score(X_train, y_train['order_filled'])
            test_score = clf.score(X_test, y_test['order_filled'])
            pred_proba = clf.predict_proba(X_test)[:, 1]
            
            try:
                roc_auc = roc_auc_score(y_test['order_filled'], pred_proba)
            except:
                roc_auc = 0
            
            print(f"    訓練準確率: {train_score:.4f}")
            print(f"    測試準確率: {test_score:.4f}")
            print(f"    ROC-AUC: {roc_auc:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        print(f"\n  最佳模型: {best_name} (準確率: {best_score:.4f})")
        
        self.models['order_filled'] = best_model
        self.results['order_filled'] = {
            'model': best_name,
            'test_accuracy': best_score,
            'roc_auc': roc_auc
        }
        
        return best_model
    
    def train_order_profitable_classifier(self, X_train, y_train, X_test, y_test):
        """
        訓練掛單盈利分類器
        """
        print("\n" + "="*70)
        print("訓練模型 2: 掛單是否盈利 (分類)")
        print("="*70)
        
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=self.random_state)
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, clf in classifiers.items():
            print(f"\n  訓練 {name}...")
            
            clf.fit(X_train, y_train['order_profitable'])
            
            train_score = clf.score(X_train, y_train['order_profitable'])
            test_score = clf.score(X_test, y_test['order_profitable'])
            
            print(f"    訓練準確率: {train_score:.4f}")
            print(f"    測試準確率: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        print(f"\n  最佳模型: {best_name} (準確率: {best_score:.4f})")
        
        self.models['order_profitable'] = best_model
        self.results['order_profitable'] = {
            'model': best_name,
            'test_accuracy': best_score
        }
        
        return best_model
    
    def train_pending_level_regressor(self, X_train, y_train, X_test, y_test):
        """
        訓練掛單點位迴歸模型
        """
        print("\n" + "="*70)
        print("訓練模型 3: 掛單點位預測 (迴歸)")
        print("="*70)
        
        regressors = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=self.random_state)
        }
        
        # 訓練買入點位
        print("\n  訓練買入掛單點位...")
        best_buy_model = None
        best_buy_r2 = -np.inf
        best_buy_name = None
        
        for name, reg in regressors.items():
            reg.fit(X_train, y_train['buy_pending_level'])
            
            train_r2 = reg.score(X_train, y_train['buy_pending_level'])
            test_r2 = reg.score(X_test, y_test['buy_pending_level'])
            test_mae = mean_absolute_error(y_test['buy_pending_level'], reg.predict(X_test))
            
            print(f"    {name}: R²={test_r2:.4f}, MAE={test_mae:.8f}")
            
            if test_r2 > best_buy_r2:
                best_buy_r2 = test_r2
                best_buy_model = reg
                best_buy_name = name
        
        # 訓練賣出點位
        print("\n  訓練賣出掛單點位...")
        best_sell_model = None
        best_sell_r2 = -np.inf
        best_sell_name = None
        
        for name, reg in regressors.items():
            reg.fit(X_train, y_train['sell_pending_level'])
            
            train_r2 = reg.score(X_train, y_train['sell_pending_level'])
            test_r2 = reg.score(X_test, y_test['sell_pending_level'])
            test_mae = mean_absolute_error(y_test['sell_pending_level'], reg.predict(X_test))
            
            print(f"    {name}: R²={test_r2:.4f}, MAE={test_mae:.8f}")
            
            if test_r2 > best_sell_r2:
                best_sell_r2 = test_r2
                best_sell_model = reg
                best_sell_name = name
        
        self.models['buy_pending_level'] = best_buy_model
        self.models['sell_pending_level'] = best_sell_model
        
        self.results['pending_levels'] = {
            'buy_model': best_buy_name,
            'buy_r2': best_buy_r2,
            'sell_model': best_sell_name,
            'sell_r2': best_sell_r2
        }
        
        return best_buy_model, best_sell_model
    
    def save_models(self, output_dir='./models'):
        """
        保存所有模型
        """
        print(f"\n保存模型到 {output_dir}...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ {model_path}")
        
        # 保存結果
        results_path = f"{output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  ✓ {results_path}")

# ============================================================================
# SECTION 4: 主執行程序
# ============================================================================

def main():
    """
    完整的BTC訓練管道
    """
    
    print("\n" + "="*70)
    print("BTC完整訓練管道")
    print("步驟: HuggingFace下載 → 計算指標 → 生成標籤 → 訓練模型")
    print("="*70)
    
    # 步驟0: 下載數據
    print("\n" + "-"*70)
    print("步驟 0: 從HuggingFace下載BTC數據")
    print("-"*70)
    
    hf_loader = HuggingFaceDataLoader()
    data = hf_loader.download_and_load()
    
    if data is None:
        print("\n✗ 數據下載失敗")
        return
    
    data = hf_loader.prepare_dataframe()
    
    # 步驟1: 計算指標
    print("\n" + "-"*70)
    print("步驟 1: 計算所有技術指標")
    print("-"*70)
    
    indicator_calc = IndicatorCalculator(data)
    data = indicator_calc.calculate_all()
    
    # 步驟2: 特徵工程
    print("\n" + "-"*70)
    print("步驟 2: 特徵工程和標籤生成")
    print("-"*70)
    
    feature_eng = FeatureEngineer(data)
    data = feature_eng.generate_labels(lookahead=5, profit_threshold=0.001)
    data = feature_eng.create_features()
    
    X, y, feature_cols, scaler = feature_eng.get_features_and_labels()
    
    # 步驟3: 數據分割
    print("\n" + "-"*70)
    print("步驟 3: 數據分割")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n訓練集: {X_train.shape[0]} 樣本")
    print(f"測試集: {X_test.shape[0]} 樣本")
    print(f"特徵數: {X_train.shape[1]}")
    
    # 步驟4: 訓練模型
    print("\n" + "-"*70)
    print("步驟 4: 訓練ML模型")
    print("-"*70)
    
    trainer = MLModelTrainer()
    trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
    trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
    trainer.train_pending_level_regressor(X_train, y_train, X_test, y_test)
    
    # 步驟5: 保存模型
    print("\n" + "-"*70)
    print("步驟 5: 保存模型")
    print("-"*70)
    
    trainer.save_models('./models')
    
    # 步驟6: 訓練完成
    print("\n" + "="*70)
    print("✓ 訓練完成！")
    print("="*70)
    print("\n下一步:")
    print("  1. 使用 real_time_prediction.py 進行實時預測")
    print("  2. 模型文件已保存到 ./models/ 目錄")
    print("  3. 訓練結果已保存到 ./models/training_results.json")
    print()
    
    return trainer, feature_eng, X_test, y_test

if __name__ == "__main__":
    trainer, feature_eng, X_test, y_test = main()
