# src/data/preprocessor.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

class DataPreprocessor:
    
    def clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Remove rows with missing essential data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Fix impossible price relationships
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        # Remove outliers (price changes > 50%)
        data['Returns'] = data['Close'].pct_change()
        outlier_mask = (data['Returns'].abs() < 0.5) | data['Returns'].isna()
        data = data[outlier_mask].copy()
        
        # Fill small gaps
        data = data.fillna(method='ffill', limit=3)
        
        return data.drop('Returns', axis=1)
    
    def normalize_returns(self, returns: pd.Series) -> pd.Series:
        return returns.fillna(0).clip(-0.2, 0.2)  # Cap extreme returns
    
    def prepare_optimization_data(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        
        returns_dict = {}
        for symbol, data in stock_data.items():
            if not data.empty and 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_dict[symbol] = self.normalize_returns(returns)
        
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.dropna()

preprocessor = DataPreprocessor()