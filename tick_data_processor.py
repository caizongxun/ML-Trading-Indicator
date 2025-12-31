"""
流段数据处理器
用途：转换、清理、优化从TradingView导出的K线氁数据
CSV格式: DateTime,Open,High,Low,Close,Volume,RSI,Stoch%K,MACD,...
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TickDataProcessor:
    """流段数据处理器"""
    
    def __init__(self, csv_path):
        """
        初始化数据处理器
        
        Args:
            csv_path (str): CSV文件路径
        """
        self.csv_path = csv_path
        self.data = None
        self.original_length = 0
        
    def load_data(self):
        """
        从TradingView CSV加载数据
        
        Returns:
            DataFrame: 加载的数据
        """
        try:
            # 第一次尝试使用标准UTF-8编码
            self.data = pd.read_csv(self.csv_path, dtype={
                'DateTime': str,
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': float,
            })
        except UnicodeDecodeError:
            # 如果失败，第二次尝试使Latin-1
            logger.warning("UTF-8 encoding failed, trying Latin-1...")
            self.data = pd.read_csv(self.csv_path, encoding='latin-1')
        
        self.original_length = len(self.data)
        logger.info(f"✓ Successfully loaded {self.original_length} rows")
        logger.info(f"  Columns: {list(self.data.columns)}")
        
        return self.data
    
    def clean_data(self):
        """
        清理数据: 移除NaN、空值、原样重複
        
        Returns:
            DataFrame: 清理后的数据
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # 上需清理前的数据
        initial_rows = len(self.data)
        
        # 1. 需移除正式列名的はNaN值
        self.data = self.data.dropna(subset=['Close', 'Volume'])
        logger.info(f"  Removed {initial_rows - len(self.data)} rows with NaN in Close/Volume")
        
        # 2. 需移除折底行 (0患型塊)
        self.data = self.data[self.data['Volume'] > 0]
        logger.info(f"  Removed {initial_rows - len(self.data)} rows with 0 volume")
        
        # 3. 需移除重複時間戳
        self.data = self.data.drop_duplicates(subset=['DateTime'], keep='last')
        logger.info(f"  Removed duplicates based on DateTime")
        
        # 4. 需推查并丢い除OHLC逆序データ
        invalid_mask = (self.data['High'] < self.data['Close']) | \
                      (self.data['High'] < self.data['Open']) | \
                      (self.data['Low'] > self.data['Close']) | \
                      (self.data['Low'] > self.data['Open'])
        
        invalid_count = invalid_mask.sum()
        self.data = self.data[~invalid_mask]
        logger.info(f"  Removed {invalid_count} rows with invalid OHLC order")
        
        # 5. 重置索引
        self.data = self.data.reset_index(drop=True)
        
        logger.info(f"✓ Data cleaning completed")
        logger.info(f"  Original: {initial_rows} rows -> Final: {len(self.data)} rows")
        logger.info(f"  Cleaned: {((initial_rows - len(self.data)) / initial_rows * 100):.2f}%")
        
        return self.data
    
    def convert_datetime(self, dt_format="%Y-%m-%d %H:%M:%S"):
        """
        转换DateTime列为pandas datetime对象
        
        Args:
            dt_format (str): 日期时間格式
            
        Returns:
            DataFrame: 会加上Datetime列
        """
        try:
            self.data['DateTime'] = pd.to_datetime(
                self.data['DateTime'], 
                format=dt_format,
                errors='coerce'
            )
            logger.info("✓ DateTime conversion successful")
        except Exception as e:
            logger.error(f"DateTime conversion failed: {e}")
            # 尝试使用自动检测
            try:
                self.data['DateTime'] = pd.to_datetime(
                    self.data['DateTime'],
                    errors='coerce'
                )
                logger.warning("使用自动检测日日时间格式")
            except Exception as e2:
                logger.error(f"Auto-detection also failed: {e2}")
        
        return self.data
    
    def validate_data(self):
        """
        验证数据质量
        
        Returns:
            dict: 验证结果报告
        """
        report = {
            'total_rows': len(self.data),
            'date_range': {
                'start': str(self.data['DateTime'].min()) if 'DateTime' in self.data.columns else 'N/A',
                'end': str(self.data['DateTime'].max()) if 'DateTime' in self.data.columns else 'N/A'
            },
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': {
                'close_min': float(self.data['Close'].min()),
                'close_max': float(self.data['Close'].max()),
                'close_mean': float(self.data['Close'].mean()),
                'volume_total': float(self.data['Volume'].sum()),
                'volume_avg': float(self.data['Volume'].mean())
            },
            'column_info': {
                'total_columns': len(self.data.columns),
                'columns': list(self.data.columns)
            }
        }
        
        logger.info("✓ Data Validation Report:")
        logger.info(f"  Total Rows: {report['total_rows']}")
        logger.info(f"  Date Range: {report['date_range']['start']} to {report['date_range']['end']}")
        logger.info(f"  Close Range: {report['statistics']['close_min']:.5f} - {report['statistics']['close_max']:.5f}")
        logger.info(f"  Total Columns: {report['column_info']['total_columns']}")
        
        return report
    
    def save_cleaned_data(self, output_path=None):
        """
        保存清理后的数据
        
        Args:
            output_path (str): 输出路径（可选）
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            # 设置默认输出路径
            output_path = str(self.csv_path).replace('.csv', '_cleaned.csv')
        
        self.data.to_csv(output_path, index=False)
        logger.info(f"✓ Cleaned data saved to: {output_path}")
        logger.info(f"  File size: {Path(output_path).stat().st_size / 1024:.2f} KB")
        
        return output_path
    
    def get_summary(self):
        """
        获取数据求一上奇重程度
        
        Returns:
            dict: 总结信息
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source_file': str(self.csv_path),
            'original_rows': self.original_length,
            'current_rows': len(self.data),
            'rows_removed': self.original_length - len(self.data),
            'removal_percentage': ((self.original_length - len(self.data)) / self.original_length * 100) if self.original_length > 0 else 0,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict()
        }
        
        return summary

def main():
    """
    主执行程序：演示如何使用数据处理器
    """
    
    print("\n" + "="*70)
    print("流段数据处理系统")
    print("="*70 + "\n")
    
    # 第1步: 输入CSV文件路径
    csv_file = input("Enter CSV file path (or press Enter for 'trading_data.csv'): ").strip()
    if not csv_file:
        csv_file = 'trading_data.csv'
    
    # 检查文件是否存在
    if not Path(csv_file).exists():
        logger.error(f"File not found: {csv_file}")
        return
    
    # 第2步: 加载数据
    processor = TickDataProcessor(csv_file)
    processor.load_data()
    
    # 第3步: 转换DateTime
    processor.convert_datetime()
    
    # 第4步: 清理数据
    processor.clean_data()
    
    # 第5步: 验证数据
    validation_report = processor.validate_data()
    
    # 第6步: 保存清理后的数据
    output_path = processor.save_cleaned_data()
    
    # 第7步: 出力求一上奇
    summary = processor.get_summary()
    
    print("\n" + "="*70)
    print("处理完成求一上奇!")
    print("="*70)
    
    print(f"\n处理结果求一上奇:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 显示条例数据
    print(f"\n\u4e2a例数据 (\u6700后5行):")
    print(processor.data.tail().to_string())
    
    return processor

if __name__ == "__main__":
    processor = main()
