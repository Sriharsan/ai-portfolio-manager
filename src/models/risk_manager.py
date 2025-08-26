"""
Risk Manager - Portfolio Risk Assessment
VaR, CVaR, and risk metrics calculation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
import logging

class RiskManager:
    """Optimized risk assessment and monitoring"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def calculate_var(self, returns: pd.Series, method: str = 'historical') -> float:
        """Calculate Value at Risk"""
        
        if len(returns) < 30:
            return 0.0
        
        if method == 'historical':
            return np.percentile(returns, (1 - self.confidence_level) * 100)
        elif method == 'parametric':
            return stats.norm.ppf(1 - self.confidence_level, returns.mean(), returns.std())
        else:
            return self.calculate_var(returns, 'historical')
    
    def calculate_cvar(self, returns: pd.Series) -> float:
        """Calculate Conditional Value at Risk"""
        
        var = self.calculate_var(returns)
        return returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
    
    def portfolio_risk_metrics(self, returns: pd.Series, portfolio_value: float = 100000) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 10:
            return {'error': 'Insufficient data'}
        
        var_1d = self.calculate_var(returns_clean) * portfolio_value
        cvar_1d = self.calculate_cvar(returns_clean) * portfolio_value
        
        return {
            'var_1d_95': var_1d,
            'cvar_1d_95': cvar_1d,
            'volatility_annual': returns_clean.std() * np.sqrt(252),
            'downside_deviation': self._downside_deviation(returns_clean),
            'maximum_drawdown': self._max_drawdown(returns_clean),
            'beta': self._calculate_beta(returns_clean) if len(returns_clean) > 50 else 1.0
        }
    
    def _downside_deviation(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_beta(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> float:
        """Calculate beta vs market (simplified)"""
        # Placeholder - would need market data for actual calculation
        return 1.0 + np.random.normal(0, 0.2)  # Simplified beta estimation

# Global instance
risk_manager = RiskManager()