"""
Performance Attribution Analysis
Analyze sources of portfolio returns
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

class AttributionAnalyzer:
    """Portfolio performance attribution"""
    
    def sector_attribution(self, portfolio_returns: Dict[str, pd.Series], 
                          weights: Dict[str, float], 
                          benchmark_returns: pd.Series) -> Dict:
        """Calculate sector-based performance attribution"""
        
        # Simplified attribution analysis
        results = {}
        
        for symbol, weight in weights.items():
            if symbol in portfolio_returns:
                asset_return = portfolio_returns[symbol].mean() * 252  # Annualized
                benchmark_return = benchmark_returns.mean() * 252
                
                excess_return = asset_return - benchmark_return
                contribution = weight * excess_return
                
                results[symbol] = {
                    'weight': weight,
                    'return': asset_return,
                    'excess_return': excess_return,
                    'contribution': contribution
                }
        
        return results
    
    def style_attribution(self, returns: pd.DataFrame) -> Dict:
        """Analyze style factors (growth vs value, size)"""
        
        # Simplified style analysis
        volatility = returns.std() * np.sqrt(252)
        
        # Classify by volatility as proxy for style
        style_classification = {}
        for column in returns.columns:
            vol = returns[column].std() * np.sqrt(252)
            if vol > volatility.median() * 1.2:
                style_classification[column] = 'growth'
            elif vol < volatility.median() * 0.8:
                style_classification[column] = 'value'
            else:
                style_classification[column] = 'blend'
        
        return style_classification

attribution_analyzer = AttributionAnalyzer()