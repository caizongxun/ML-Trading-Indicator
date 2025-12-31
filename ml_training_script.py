"""
ML Trading Data Exporter & Model Trainer
用途：從 TradingView 導出指標數據，訓練機器学習模型預測掛單點位有效性
版本：1.0
要求：Python 3.8+, pandas, scikit-learn, matplotlib
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns

class MLDataHandler:
    """Handling trading data loading, cleaning and feature engineering"""
    
    def __init__(self, csv_path=None):
        """
        Initialize data handler
        
        Args:
            csv_path (str): CSV file path (exported from TradingView)
        """
        self.csv_path = csv_path
        self.data = None
        self.scaler_features = StandardScaler()
        self.scaler_target = MinMaxScaler()
        
    def load_data(self, csv_path=None):
        """Load CSV data"""
        if csv_path:
            self.csv_path = csv_path
        
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Successfully loaded {len(self.data)} candles")
            print(f"  Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False
    
    def create_sample_data(self, n_samples=1000):
        """
        Generate sample data for testing
        
        Args:
            n_samples (int): Number of samples to generate
        """
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='15T')
        
        rsi = np.random.uniform(20, 80, n_samples)
        stoch = np.random.uniform(10, 90, n_samples)
        macd = np.random.uniform(-1, 1, n_samples)
        bb_width = np.random.uniform(0.5, 3.0, n_samples)
        
        momentum = np.random.uniform(-100, 100, n_samples)
        volatility = np.random.uniform(0, 100, n_samples)
        rsi_convergence = np.random.uniform(0, 100, n_samples)
        composite = np.random.uniform(-100, 100, n_samples)
        
        close_price = np.random.uniform(1.0500, 1.1000, n_samples)
        buy_pending = close_price - np.random.uniform(0.001, 0.010, n_samples)
        sell_pending = close_price + np.random.uniform(0.001, 0.010, n_samples)
        
        order_filled = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        order_profitable = np.where(
            (order_filled == 1) & (momentum > 0), 1, 
            np.where((order_filled == 1) & (momentum < 0), np.random.choice([0, 1]), 0)
        )
        
        self.data = pd.DataFrame({
            'datetime': dates,
            'rsi': rsi,
            'stoch': stoch,
            'macd': macd,
            'bb_width': bb_width,
            'momentum_score': momentum,
            'volatility_index': volatility,
            'rsi_convergence': rsi_convergence,
            'composite_signal': composite,
            'close_price': close_price,
            'buy_pending_level': buy_pending,
            'sell_pending_level': sell_pending,
            'order_filled': order_filled,
            'order_profitable': order_profitable
        })
        
        print(f"Generated {n_samples} sample data records")
        return self.data
    
    def preprocess_data(self):
        """Data cleaning and preprocessing"""
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        print(f"  Removed NaN values: {initial_rows - len(self.data)} rows")
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < Q1 - 1.5 * IQR) | 
                       (self.data[col] > Q3 + 1.5 * IQR))
            self.data = self.data[~outliers]
        
        print(f"Data cleaning complete, retained {len(self.data)} rows")
        return self.data
    
    def feature_engineering(self):
        """Feature engineering: create new features"""
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
        
        print("Feature engineering complete")
        return self.data
    
    def prepare_ml_data(self):
        """Prepare features and labels for ML training"""
        feature_cols = [
            'rsi', 'stoch', 'macd', 'bb_width', 
            'momentum_score', 'volatility_index', 'rsi_convergence', 
            'composite_signal', 'momentum_change', 'rsi_slope',
            'volatility_ratio', 'price_to_buy_distance', 
            'price_to_sell_distance', 'order_fill_rate', 'order_profit_rate'
        ]
        
        target_cols = ['order_filled', 'order_profitable']
        
        X = self.data[feature_cols].copy()
        y = self.data[target_cols].copy()
        
        X_scaled = self.scaler_features.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        print(f"Preparation complete: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y, feature_cols

class MLModelTrainer:
    """Machine learning model trainer"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_order_filled_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train order fill probability classification model
        
        Predict: Will the pending order be triggered
        """
        print("\n" + "="*60)
        print("Training Model 1: Order Fill Probability (Classification)")
        print("="*60)
        
        classifiers = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, clf in classifiers.items():
            print(f"\n  Training {name}...")
            
            clf.fit(X_train, y_train['order_filled'])
            
            train_score = clf.score(X_train, y_train['order_filled'])
            test_score = clf.score(X_test, y_test['order_filled'])
            pred = clf.predict(X_test)
            pred_proba = clf.predict_proba(X_test)[:, 1]
            
            try:
                roc_auc = roc_auc_score(y_test['order_filled'], pred_proba)
            except:
                roc_auc = 0
            
            print(f"    Train Accuracy: {train_score:.4f}")
            print(f"    Test Accuracy: {test_score:.4f}")
            print(f"    ROC-AUC: {roc_auc:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        print(f"\n  Best Model: {best_name} (Accuracy: {best_score:.4f})")
        
        self.models['order_filled'] = best_model
        self.results['order_filled'] = {
            'model': best_name,
            'test_accuracy': best_score,
            'feature_importance': None
        }
        
        if hasattr(best_model, 'feature_importances_'):
            self.results['order_filled']['feature_importance'] = best_model.feature_importances_
        
        return best_model
    
    def train_order_profitable_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train order profit probability classification model
        
        Predict: Will the order be profitable
        """
        print("\n" + "="*60)
        print("Training Model 2: Order Profit Probability (Classification)")
        print("="*60)
        
        classifiers = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, clf in classifiers.items():
            print(f"\n  Training {name}...")
            
            clf.fit(X_train, y_train['order_profitable'])
            
            train_score = clf.score(X_train, y_train['order_profitable'])
            test_score = clf.score(X_test, y_test['order_profitable'])
            
            print(f"    Train Accuracy: {train_score:.4f}")
            print(f"    Test Accuracy: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = clf
                best_name = name
        
        print(f"\n  Best Model: {best_name} (Accuracy: {best_score:.4f})")
        
        self.models['order_profitable'] = best_model
        self.results['order_profitable'] = {
            'model': best_name,
            'test_accuracy': best_score,
            'feature_importance': None
        }
        
        if hasattr(best_model, 'feature_importances_'):
            self.results['order_profitable']['feature_importance'] = best_model.feature_importances_
        
        return best_model
    
    def train_pending_level_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train pending order level prediction model
        
        Predict: Optimal buy/sell pending order levels
        """
        print("\n" + "="*60)
        print("Training Model 3: Pending Order Level Prediction (Regression)")
        print("="*60)
        
        regressors = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        print("\n  Training buy pending level...")
        best_buy_model = None
        best_buy_r2 = -np.inf
        
        for name, reg in regressors.items():
            reg.fit(X_train, y_train['buy_pending_level'])
            
            train_r2 = reg.score(X_train, y_train['buy_pending_level'])
            test_r2 = reg.score(X_test, y_test['buy_pending_level'])
            test_mae = mean_absolute_error(y_test['buy_pending_level'], 
                                          reg.predict(X_test))
            
            print(f"    {name}:")
            print(f"      Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            print(f"      MAE: {test_mae:.6f}")
            
            if test_r2 > best_buy_r2:
                best_buy_r2 = test_r2
                best_buy_model = reg
                best_buy_name = name
        
        print("\n  Training sell pending level...")
        best_sell_model = None
        best_sell_r2 = -np.inf
        
        for name, reg in regressors.items():
            reg.fit(X_train, y_train['sell_pending_level'])
            
            train_r2 = reg.score(X_train, y_train['sell_pending_level'])
            test_r2 = reg.score(X_test, y_test['sell_pending_level'])
            test_mae = mean_absolute_error(y_test['sell_pending_level'], 
                                          reg.predict(X_test))
            
            print(f"    {name}:")
            print(f"      Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            print(f"      MAE: {test_mae:.6f}")
            
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
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("Model Evaluation Summary")
        print("="*60)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, model in self.models.items():
            if 'pending_level' not in model_name:
                pred = model.predict(X_test)
                pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                accuracy = model.score(X_test, y_test[model_name])
                
                print(f"\n{model_name}: Accuracy {accuracy:.4f}")
                
                summary['models'][model_name] = {
                    'accuracy': float(accuracy),
                    'test_samples': len(X_test)
                }
        
        return summary
    
    def save_models(self, output_dir='./models'):
        """Save trained models"""
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved: {model_path}")
        
        results_path = f"{output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved: {results_path}")
    
    def load_models(self, model_dir='./models'):
        """Load trained models"""
        model_files = Path(model_dir).glob('*_model.pkl')
        
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                model_name = model_file.stem.replace('_model', '')
                self.models[model_name] = model
                print(f"Model loaded: {model_name}")

class OrderPredictor:
    """Pending order predictor based on trained models"""
    
    def __init__(self, trainer, feature_cols):
        self.trainer = trainer
        self.feature_cols = feature_cols
    
    def predict_order_signal(self, current_features):
        """
        Predict if pending order should be placed
        
        Args:
            current_features (dict): Current indicator values dictionary
        
        Returns:
            dict: Prediction results
        """
        feature_vector = np.array([
            current_features.get(col, 0) for col in self.feature_cols
        ]).reshape(1, -1)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'order_fill_probability': 0.0,
            'order_profit_probability': 0.0,
            'buy_pending_level': None,
            'sell_pending_level': None,
            'recommendation': 'HOLD'
        }
        
        if 'order_filled' in self.trainer.models:
            model = self.trainer.models['order_filled']
            result['order_fill_probability'] = float(
                model.predict_proba(feature_vector)[0][1]
            )
        
        if 'order_profitable' in self.trainer.models:
            model = self.trainer.models['order_profitable']
            result['order_profit_probability'] = float(
                model.predict_proba(feature_vector)[0][1]
            )
        
        if 'buy_pending_level' in self.trainer.models:
            result['buy_pending_level'] = float(
                self.trainer.models['buy_pending_level'].predict(feature_vector)[0]
            )
        
        if 'sell_pending_level' in self.trainer.models:
            result['sell_pending_level'] = float(
                self.trainer.models['sell_pending_level'].predict(feature_vector)[0]
            )
        
        fill_prob = result['order_fill_probability']
        profit_prob = result['order_profit_probability']
        
        if fill_prob > 0.7 and profit_prob > 0.6:
            result['recommendation'] = 'STRONG_BUY'
        elif fill_prob > 0.6 and profit_prob > 0.5:
            result['recommendation'] = 'BUY'
        elif fill_prob > 0.4 and profit_prob > 0.5:
            result['recommendation'] = 'WATCH'
        else:
            result['recommendation'] = 'HOLD'
        
        return result

def main():
    """Main training workflow"""
    
    print("\n" + "="*60)
    print("ML Trading Data Processing and Model Training System")
    print("="*60 + "\n")
    
    print("Step 1: Data Loading and Preparation")
    print("-" * 60)
    
    handler = MLDataHandler()
    handler.create_sample_data(n_samples=1500)
    handler.preprocess_data()
    handler.feature_engineering()
    
    print("\nStep 2: Prepare ML Data")
    print("-" * 60)
    
    X, y, feature_cols = handler.prepare_ml_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nStep 3: Model Training")
    print("-" * 60)
    
    trainer = MLModelTrainer()
    
    trainer.train_order_filled_classifier(X_train, y_train, X_test, y_test)
    trainer.train_order_profitable_classifier(X_train, y_train, X_test, y_test)
    trainer.train_pending_level_regressor(X_train, y_train, X_test, y_test)
    
    print("\nStep 4: Model Evaluation")
    print("-" * 60)
    
    trainer.evaluate_all_models(X_test, y_test)
    
    print("\nStep 5: Save Models")
    print("-" * 60)
    
    trainer.save_models()
    
    print("\nStep 6: Test Real-time Prediction")
    print("-" * 60)
    
    predictor = OrderPredictor(trainer, feature_cols)
    
    test_sample = X_test.iloc[0].to_dict()
    prediction = predictor.predict_order_signal(test_sample)
    
    print("\nPrediction Result Example:")
    print(json.dumps(prediction, indent=2))
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    
    return trainer, handler, predictor

if __name__ == "__main__":
    trainer, handler, predictor = main()
