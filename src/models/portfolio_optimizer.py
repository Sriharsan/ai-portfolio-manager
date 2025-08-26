"""
Portfolio Optimizer - Modern Portfolio Theory Implementation
Efficient frontier calculation with risk management
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import logging

class PortfolioOptimizer:
    """Efficient portfolio optimization using MPT"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def optimize_portfolio(
        self, 
        returns: pd.DataFrame,
        method: str = 'max_sharpe',
        target_return: Optional[float] = None
    ) -> Dict:
        """
        Optimize portfolio allocation
        
        Args:
            returns: DataFrame of asset returns
            method: 'max_sharpe', 'min_volatility', 'target_return'
            target_return: Required return (for target_return method)
        
        Returns:
            Optimization results with weights and metrics
        """
        
        # Calculate expected returns and covariance matrix
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252     # Annualized
        n_assets = len(mean_returns)
        
        # Optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 0.2) for _ in range(n_assets))  # Max 20% per asset
        
        if method == 'max_sharpe':
            result = self._maximize_sharpe_ratio(mean_returns, cov_matrix, bounds, constraints)
        elif method == 'min_volatility':
            result = self._minimize_volatility(mean_returns, cov_matrix, bounds, constraints)
        elif method == 'target_return' and target_return:
            constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})
            result = self._minimize_volatility(mean_returns, cov_matrix, bounds, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate portfolio metrics
        weights = result.x
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': dict(zip(returns.columns, weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result.success
        }
    
    def _maximize_sharpe_ratio(self, mean_returns, cov_matrix, bounds, constraints):
        """Maximize Sharpe ratio optimization"""
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Initial guess: equal weights
        initial_guess = np.array([1/len(mean_returns)] * len(mean_returns))
        
        return minimize(
            negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    def _minimize_volatility(self, mean_returns, cov_matrix, bounds, constraints):
        """Minimize volatility optimization"""
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/len(mean_returns)] * len(mean_returns))
        
        return minimize(
            portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    def efficient_frontier(self, returns: pd.DataFrame, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            returns: Asset returns DataFrame
            num_portfolios: Number of portfolios to generate
        
        Returns:
            DataFrame with efficient frontier portfolios
        """
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        min_return = mean_returns.min()
        max_return = mean_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target in target_returns:
            try:
                result = self.optimize_portfolio(returns, 'target_return', target)
                if result['optimization_success']:
                    efficient_portfolios.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio']
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)

# Global optimizer instance
optimizer = PortfolioOptimizer()