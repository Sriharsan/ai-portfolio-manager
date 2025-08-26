"""
Portfolio Performance Analytics - Core Metrics Calculation
Efficient computation of key performance indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

class PerformanceAnalyzer:
    """Optimized portfolio performance analytics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Portfolio returns series
            benchmark: Benchmark returns (optional)
        
        Returns:
            Dictionary of performance metrics
        """
        
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        volatility = returns.std() * np.sqrt(self.trading_days)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'win_rate': (returns > 0).mean(),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }
        
        # Benchmark comparison
        if benchmark is not None:
            benchmark = benchmark.reindex(returns.index).dropna()
            if len(benchmark) > 0:
                benchmark_return = (1 + benchmark).prod() - 1
                excess_return = total_return - benchmark_return
                tracking_error = (returns - benchmark).std() * np.sqrt(self.trading_days)
                
                metrics.update({
                    'benchmark_return': benchmark_return,
                    'excess_return': excess_return,
                    'information_ratio': excess_return / tracking_error if tracking_error > 0 else 0,
                    'beta': self._calculate_beta(returns, benchmark)
                })
        
        return metrics
    
    def _calculate_beta(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate portfolio beta vs benchmark"""
        
        aligned_data = pd.DataFrame({'portfolio': returns, 'benchmark': benchmark}).dropna()
        
        if len(aligned_data) < 10:
            return 1.0
        
        covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
        benchmark_variance = aligned_data['benchmark'].var()
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def rolling_metrics(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        rolling_data = pd.DataFrame(index=returns.index)
        rolling_data['rolling_return'] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_data['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(self.trading_days)
        rolling_data['rolling_sharpe'] = (rolling_data['rolling_return'] - self.risk_free_rate) / rolling_data['rolling_volatility']
        
        return rolling_data.dropna()

# Global instance
performance_analyzer = PerformanceAnalyzer()