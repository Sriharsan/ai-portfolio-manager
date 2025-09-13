# src/models/ml_engine.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

class MachineLearningEngine:
    """Advanced ML engine for return prediction and market analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical and fundamental features"""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns_1d'] = data['Close'].pct_change()
        features['returns_5d'] = data['Close'].pct_change(5)
        features['returns_20d'] = data['Close'].pct_change(20)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(data) >= window:
                features[f'sma_{window}'] = data['Close'].rolling(window).mean()
                features[f'price_to_sma_{window}'] = data['Close'] / features[f'sma_{window}']
        
        # Volatility features
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Volume features
        features['volume_sma_10'] = data['Volume'].rolling(10).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma_10']
        
        # Technical indicators
        if 'RSI' in data.columns:
            features['rsi'] = data['RSI']
            features['rsi_oversold'] = (data['RSI'] < 30).astype(int)
            features['rsi_overbought'] = (data['RSI'] > 70).astype(int)
        
        # MACD features
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            features['macd'] = data['MACD']
            features['macd_signal'] = data['MACD_Signal']
            features['macd_histogram'] = data['MACD_Histogram'] if 'MACD_Histogram' in data.columns else 0
        
        # Momentum features
        features['momentum_10d'] = data['Close'] / data['Close'].shift(10) - 1
        features['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        return features.dropna()
    
    def lstm_return_prediction(self, data: pd.DataFrame, 
                             symbol: str,
                             prediction_days: int = 5,
                             sequence_length: int = 20) -> Dict:
        """LSTM-based return prediction"""
        try:
            # Create features
            features = self.create_features(data)
            
            if len(features) < sequence_length + prediction_days + 50:
                return {'error': 'Insufficient data for LSTM training'}
            
            # Prepare target variable (future returns)
            features['target'] = features['returns_1d'].shift(-prediction_days)
            
            # Remove NaN values
            model_data = features.dropna()
            
            # Select feature columns
            feature_cols = [col for col in model_data.columns if col != 'target']
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(model_data[feature_cols])
            
            # Create sequences for LSTM
            X, y = [], []
            for i in range(sequence_length, len(scaled_features)):
                X.append(scaled_features[i-sequence_length:i])
                y.append(model_data['target'].iloc[i])
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 50:
                return {'error': 'Insufficient sequences for LSTM training'}
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(X_train, y_train, 
                              batch_size=32, epochs=50, 
                              validation_data=(X_test, y_test),
                              verbose=0)
            
            # Make predictions
            y_pred = model.predict(X_test, verbose=0)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Predict next period
            last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            next_prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Store model and scaler
            self.models[f'{symbol}_lstm'] = model
            self.scalers[f'{symbol}_lstm'] = scaler
            
            return {
                'model_type': 'LSTM',
                'prediction': float(next_prediction),
                'mse': float(mse),
                'r2_score': float(r2),
                'training_loss': history.history['loss'][-1],
                'validation_loss': history.history['val_loss'][-1],
                'prediction_horizon_days': prediction_days,
                'model_status': 'trained'
            }
            
        except Exception as e:
            self.logger.error(f"LSTM prediction failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def ensemble_forecasting(self, data: pd.DataFrame, 
                           symbol: str) -> Dict:
        """Ensemble model combining multiple ML algorithms"""
        try:
            features = self.create_features(data)
            
            if len(features) < 100:
                return {'error': 'Insufficient data for ensemble modeling'}
            
            # Prepare target (next day return)
            features['target'] = features['returns_1d'].shift(-1)
            model_data = features.dropna()
            
            # Select features
            feature_cols = [col for col in model_data.columns 
                          if col not in ['target', 'returns_1d']]
            
            X = model_data[feature_cols]
            y = model_data['target']
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            predictions = {}
            model_scores = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Score model
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                predictions[name] = y_pred
                model_scores[name] = r2
                
                # Predict next value
                next_pred = model.predict(X_train_scaled[-1:].reshape(1, -1))[0]
                predictions[f'{name}_next'] = next_pred
            
            # Ensemble prediction (weighted by R² scores)
            total_score = sum(max(0, score) for score in model_scores.values())
            if total_score > 0:
                weights = {name: max(0, score) / total_score 
                          for name, score in model_scores.items()}
            else:
                weights = {name: 1/len(model_scores) for name in model_scores}
            
            ensemble_next = sum(weights[name] * predictions[f'{name}_next'] 
                              for name in models.keys())
            
            # Store models
            self.models[f'{symbol}_ensemble'] = models
            self.scalers[f'{symbol}_ensemble'] = scaler
            
            return {
                'model_type': 'ensemble',
                'prediction': float(ensemble_next),
                'individual_predictions': {name: float(predictions[f'{name}_next']) 
                                         for name in models.keys()},
                'model_weights': weights,
                'model_scores': model_scores,
                'ensemble_r2': np.mean(list(model_scores.values())),
                'feature_importance': self._get_feature_importance(models['rf'], feature_cols),
                'model_status': 'trained'
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble forecasting failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def regime_detection(self, returns: pd.Series) -> Dict:
        """Market regime detection using volatility clustering"""
        try:
            # Calculate rolling volatility
            vol_window = 20
            rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
            
            # Define regime thresholds
            vol_median = rolling_vol.median()
            low_vol_threshold = vol_median * 0.75
            high_vol_threshold = vol_median * 1.25
            
            # Classify regimes
            regimes = pd.Series(index=returns.index, dtype='category')
            regimes[rolling_vol < low_vol_threshold] = 'low_volatility'
            regimes[rolling_vol > high_vol_threshold] = 'high_volatility'
            regimes[regimes.isna()] = 'normal'
            
            # Current regime
            current_regime = regimes.iloc[-1] if not regimes.empty else 'normal'
            
            # Regime statistics
            regime_stats = {}
            for regime in ['low_volatility', 'normal', 'high_volatility']:
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    regime_returns = returns[regime_mask]
                    regime_stats[regime] = {
                        'mean_return': regime_returns.mean() * 252,
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'frequency': regime_mask.mean(),
                        'sharpe_ratio': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252))
                    }
            
            # Regime persistence
            regime_changes = (regimes != regimes.shift()).sum()
            avg_regime_duration = len(regimes) / regime_changes if regime_changes > 0 else len(regimes)
            
            return {
                'current_regime': current_regime,
                'regime_stats': regime_stats,
                'avg_regime_duration': avg_regime_duration,
                'regime_transitions': regime_changes,
                'volatility_thresholds': {
                    'low': low_vol_threshold,
                    'high': high_vol_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return {'error': str(e)}
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                # Sort by importance
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Feature importance extraction failed: {e}")
            return {}

# Global instance
ml_engine = MachineLearningEngine()