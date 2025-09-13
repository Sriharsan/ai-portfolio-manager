# src/models/enhanced_portfolio_optimizer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from .advanced_optimization import institutional_optimizer
from .institutional_risk import institutional_risk_manager
from .ml_engine import ml_engine

class EnhancedPortfolioOptimizer:
    """Integration layer for all optimization methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimizer = institutional_optimizer
        self.risk_manager = institutional_risk_manager
        self.ml_engine = ml_engine
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                         method: str = 'markowitz',
                         constraints: Optional[Dict] = None,
                         **kwargs) -> Dict:
        """Unified portfolio optimization interface"""
        try:
            if method == 'markowitz':
                return self.optimizer.markowitz_optimization(returns, **kwargs)
            elif method == 'black_litterman':
                return self.optimizer.black_litterman_optimization(returns, **kwargs)
            elif method == 'risk_parity':
                return self.optimizer.risk_parity_optimization(returns)
            elif method == 'ml_enhanced':
                return self._ml_enhanced_optimization(returns, **kwargs)
            else:
                return {'error': f'Unknown optimization method: {method}'}
                
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {'error': str(e)}
    
    def _ml_enhanced_optimization(self, returns: pd.DataFrame, **kwargs) -> Dict:
        """ML-enhanced optimization combining predictions with traditional methods"""
        try:
            # Get ML predictions for each asset
            predictions = {}
            for symbol in returns.columns:
                pred_result = self.ml_engine.ensemble_forecasting(
                    pd.DataFrame({'Close': returns[symbol].cumsum()}, index=returns.index),
                    symbol
                )
                if 'prediction' in pred_result:
                    predictions[symbol] = pred_result['prediction']
            
            if not predictions:
                # Fallback to traditional optimization
                return self.optimizer.markowitz_optimization(returns)
            
            # Adjust expected returns based on ML predictions
            ml_returns = returns.copy()
            for symbol, prediction in predictions.items():
                if symbol in ml_returns.columns:
                    # Blend historical mean with ML prediction
                    historical_mean = ml_returns[symbol].mean()
                    blended_return = 0.7 * historical_mean + 0.3 * prediction
                    ml_returns[symbol] = ml_returns[symbol] + (blended_return - historical_mean)
            
            # Optimize with ML-adjusted returns
            return self.optimizer.markowitz_optimization(ml_returns, **kwargs)
            
        except Exception as e:
            self.logger.error(f"ML-enhanced optimization failed: {e}")
            return self.optimizer.markowitz_optimization(returns, **kwargs)

# Global instance
enhanced_optimizer = EnhancedPortfolioOptimizer()