"""
高级ML指标Python实现
用于从TradingView导出的数据中计算3个新的高级指标
用途：用这些指标作为ML模型的新特征

3个指标：
1. LII - 流动性失衡指数 (Liquidity Imbalance Index)
2. MTRI - 多时间框架共鸣指数 (Multi-Timeframe Resonance Index)  
3. DSRFI - 动态支撑阻力破裂指数 (Dynamic S/R Fractal Index)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

class AdvancedIndicators:
    """计算3个高级ML指标"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化高级指标计算器
        
        Args:
            data: 包含OHLCV数据的DataFrame
                  必需列: high, low, close, volume
        """
        self.data = data.copy()
        self.data = self.data.reset_index(drop=True)
        
    def calculate_lii(self) -> pd.Series:
        """
        计算流动性失衡指数 (LII)
        
        LII检测市场中的流动性空隙和供需失衡
        
        计算逻辑：
        - LII = (高-低) / (高+低) * (成交量 / 20周期均成交量)
        - LII > 75: 严重失衡向上（供给不足）
        - LII < 25: 严重失衡向下（需求不足）
        - 45-55: 平衡状态
        
        Returns:
            Series: LII值 (0-100)
        """
        high = self.data['high'].values
        low = self.data['low'].values
        volume = self.data['volume'].values
        
        # 计算范围比率
        price_range = high - low
        price_sum = high + low + 0.001  # 避免除以0
        range_ratio = price_range / price_sum
        
        # 计算20周期平均成交量
        vol_sma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 0.001)
        
        # 综合LII
        lii_raw = range_ratio * vol_ratio
        lii = np.minimum(100, lii_raw * 100)
        
        return pd.Series(lii, index=self.data.index)
    
    def calculate_mtri(self) -> pd.Series:
        """
        计算多时间框架共鸣指数 (MTRI)
        
        MTRI检测不同指标之间的共鸣，识别高确定性交易机会
        
        计算逻辑：
        1. RSI共鸣度 = |RSI - 50| / 50
        2. MACD共鸣度 = |MACD - Signal| / ATR
        3. 动量共鸣度 = |ROC(12)| * 系数
        MTRI = (RSI共鸣 + MACD共鸣 + 动量共鸣) / 3 * 100
        
        Returns:
            Series: MTRI值 (0-100)
        """
        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values
        
        # 1. 计算RSI共鸣度
        rsi_series = self._calculate_rsi(close, period=14)
        rsi_resonance = np.abs(rsi_series - 50) / 50
        
        # 2. 计算MACD共鸣度
        macd_line, signal_line = self._calculate_macd(close)
        atr_val = self._calculate_atr(high, low, close, period=14)
        macd_diff = np.abs(macd_line - signal_line)
        macd_resonance = np.minimum(1.0, (macd_diff / (atr_val + 0.001)) * 0.5)
        
        # 3. 计算动量共鸣度 (ROC 12周期)
        roc_val = np.zeros_like(close)
        for i in range(12, len(close)):
            roc_val[i] = (close[i] - close[i-12]) / close[i-12]
        momentum_resonance = np.minimum(1.0, np.abs(roc_val) * 10)
        
        # 综合MTRI
        mtri_raw = (rsi_resonance + macd_resonance + momentum_resonance) / 3
        mtri = mtri_raw * 100
        
        return pd.Series(mtri, index=self.data.index)
    
    def calculate_dsrfi(self) -> pd.Series:
        """
        计算动态支撑阻力破裂指数 (DSRFI)
        
        DSRFI使用分形几何学原理识别价格突破关键支撑/阻力的时刻
        
        计算逻辑：
        1. 历史形态匹配度 = 1 - (价格差异 + 时间差异) / 2
        2. 破裂强度指数 = (ATR / 平均ATR) * 突破幅度
        3. 成交量确认 = 当前成交量 / 20周期均成交量
        DSRFI = 形态匹配 * 破裂强度 * 成交量 * 100
        
        Returns:
            Series: DSRFI值 (0-100)
        """
        high = self.data['high'].values
        low = self.data['low'].values
        close = self.data['close'].values
        volume = self.data['volume'].values
        
        # 1. 历史形态匹配度
        recent_range = high - low
        recent_close_ratio = (close - low) / (high - low + 0.001)
        
        hist_avg_range = pd.Series(recent_range).rolling(window=50, min_periods=1).mean().values
        hist_avg_close_ratio = pd.Series(recent_close_ratio).rolling(window=50, min_periods=1).mean().values
        
        range_diff = np.abs(recent_range - hist_avg_range) / (hist_avg_range + 0.001)
        ratio_diff = np.abs(recent_close_ratio - hist_avg_close_ratio)
        pattern_similarity = np.maximum(0, 1.0 - (range_diff + ratio_diff) / 2)
        
        # 2. 破裂强度指数
        atr_val = self._calculate_atr(high, low, close, period=14)
        atr_sma = pd.Series(atr_val).rolling(window=50, min_periods=1).mean().values
        
        midline = (high + np.roll(high, 1) + low + np.roll(low, 1)) / 4
        midline[0] = (high[0] + low[0]) / 2
        
        price_diff = np.abs(close - midline) / (high - low + 0.001)
        atr_ratio = atr_val / (atr_sma + 0.001)
        breakout_strength = np.minimum(1.0, atr_ratio * price_diff)
        
        # 3. 成交量确认
        vol_sma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
        vol_confirm = np.minimum(1.0, volume / (vol_sma + 0.001))
        
        # 综合DSRFI
        dsrfi_raw = pattern_similarity * breakout_strength * vol_confirm
        dsrfi = dsrfi_raw * 100
        
        return pd.Series(dsrfi, index=self.data.index)
    
    def add_all_advanced_indicators(self) -> pd.DataFrame:
        """
        计算所有3个高级指标并添加到数据中
        
        Returns:
            DataFrame: 添加了LII, MTRI, DSRFI列的数据
        """
        self.data['lii'] = self.calculate_lii()
        self.data['mtri'] = self.calculate_mtri()
        self.data['dsrfi'] = self.calculate_dsrfi()
        
        return self.data
    
    # ========== 辅助函数 ==========
    
    @staticmethod
    def _calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close, dtype=float)
        avg_loss = np.zeros_like(close, dtype=float)
        
        avg_gain[period] = gain[:period].mean()
        avg_loss[period] = loss[:period].mean()
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = avg_gain / (avg_loss + 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """计算MACD"""
        close_series = pd.Series(close)
        ema_fast = close_series.ewm(span=fast).mean().values
        ema_slow = close_series.ewm(span=slow).mean().values
        
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        
        return macd_line, signal_line
    
    @staticmethod
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算ATR"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        
        return atr


def main():
    """示例使用"""
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'high': np.random.uniform(1.0600, 1.1000, n_samples),
        'low': np.random.uniform(1.0500, 1.0900, n_samples),
        'close': np.random.uniform(1.0550, 1.0950, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # 确保high >= close >= low
    for i in range(len(data)):
        sorted_prices = sorted([data.loc[i, 'high'], data.loc[i, 'close'], data.loc[i, 'low']])
        data.loc[i, 'low'] = sorted_prices[0]
        data.loc[i, 'close'] = sorted_prices[1]
        data.loc[i, 'high'] = sorted_prices[2]
    
    # 计算指标
    indicator = AdvancedIndicators(data)
    result = indicator.add_all_advanced_indicators()
    
    print("\n" + "="*70)
    print("高级ML指标计算结果")
    print("="*70)
    print(f"\n总样本数: {len(result)}")
    print(f"\n最后10行数据:")
    print(result[['high', 'low', 'close', 'volume', 'lii', 'mtri', 'dsrfi']].tail(10))
    
    print(f"\n\n指标统计:")
    print(f"\nLII (流动性失衡指数):")
    print(f"  最小值: {result['lii'].min():.2f}")
    print(f"  最大值: {result['lii'].max():.2f}")
    print(f"  均值: {result['lii'].mean():.2f}")
    print(f"  标准差: {result['lii'].std():.2f}")
    
    print(f"\nMTRI (多时间框架共鸣指数):")
    print(f"  最小值: {result['mtri'].min():.2f}")
    print(f"  最大值: {result['mtri'].max():.2f}")
    print(f"  均值: {result['mtri'].mean():.2f}")
    print(f"  标准差: {result['mtri'].std():.2f}")
    
    print(f"\nDSRFI (动态支撑阻力破裂指数):")
    print(f"  最小值: {result['dsrfi'].min():.2f}")
    print(f"  最大值: {result['dsrfi'].max():.2f}")
    print(f"  均值: {result['dsrfi'].mean():.2f}")
    print(f"  标准差: {result['dsrfi'].std():.2f}")
    
    print(f"\n\n相关性分析:")
    corr_matrix = result[['lii', 'mtri', 'dsrfi']].corr()
    print(corr_matrix)
    
    return result

if __name__ == "__main__":
    result_df = main()
