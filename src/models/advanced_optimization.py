# src/models/advanced_optimization.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
import logging

class InstitutionalOptimizer:
    """BlackRock-level portfolio optimization with multiple algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def markowitz_optimization(self, returns: pd.DataFrame, 
                             target_return: Optional[float] = None,
                             risk_aversion: float = 1.0) -> Dict:
        """Mean-variance optimization using quadratic programming"""
        try:
            n_assets = len(returns.columns)
            mu = returns.mean() * 252  # Annualized returns
            Sigma = returns.cov() * 252  # Annualized covariance
            
            # Define optimization variables
            w = cp.Variable(n_assets)
            
            # Objective: maximize return - risk_aversion * variance
            portfolio_return = mu.T @ w
            portfolio_variance = cp.quad_form(w, Sigma.values)
            
            if target_return:
                # Target return optimization
                objective = cp.Minimize(portfolio_variance)
                constraints = [
                    cp.sum(w) == 1,
                    w >= 0,
                    portfolio_return >= target_return
                ]
            else:
                # Max Sharpe-like optimization
                objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
                constraints = [cp.sum(w) == 1, w >= 0]
            
            # Solve optimization
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            if prob.status == 'optimal':
                weights = dict(zip(returns.columns, w.value))
                expected_return = float(portfolio_return.value)
                expected_vol = float(np.sqrt(portfolio_variance.value))
                
                return {
                    'weights': weights,
                    'expected_return': expected_return,
                    'expected_volatility': expected_vol,
                    'sharpe_ratio': expected_return / expected_vol if expected_vol > 0 else 0,
                    'optimization_status': 'optimal'
                }
            else:
                return {'optimization_status': 'failed', 'error': prob.status}
                
        except Exception as e:
            self.logger.error(f"Markowitz optimization failed: {e}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def black_litterman_optimization(self, returns: pd.DataFrame,
                                   market_caps: Dict[str, float],
                                   views: Dict[str, float],
                                   tau: float = 0.025) -> Dict:
        """Black-Litterman model with investor views"""
        try:
            symbols = returns.columns.tolist()
            mu_market = returns.mean() * 252
            Sigma = returns.cov() * 252
            
            # Market capitalization weights
            total_mcap = sum(market_caps.values())
            w_market = np.array([market_caps.get(s, total_mcap/len(symbols))/total_mcap for s in symbols])
            
            # Implied equilibrium returns
            risk_aversion = 3.0  # Typical value
            pi = risk_aversion * Sigma.values @ w_market
            
            # Incorporate views
            if views:
                P = np.zeros((len(views), len(symbols)))
                Q = np.zeros(len(views))
                
                for i, (symbol, view) in enumerate(views.items()):
                    if symbol in symbols:
                        P[i, symbols.index(symbol)] = 1.0
                        Q[i] = view
                
                # View uncertainty (simplified)
                Omega = np.eye(len(views)) * 0.01
                
                # Black-Litterman formula
                M1 = np.linalg.inv(tau * Sigma.values)
                M2 = P.T @ np.linalg.inv(Omega) @ P
                M3 = np.linalg.inv(tau * Sigma.values) @ pi
                M4 = P.T @ np.linalg.inv(Omega) @ Q
                
                # New expected returns
                mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
            else:
                mu_bl = pi
            
            # Optimize with Black-Litterman returns
            bl_returns = pd.Series(mu_bl, index=symbols)
            
            # Convert to dataframe for optimization
            returns_df = pd.DataFrame(index=[0])
            for symbol in symbols:
                returns_df[symbol] = [bl_returns[symbol] / 252]
            
            return self.markowitz_optimization(returns_df.iloc[:1])
            
        except Exception as e:
            self.logger.error(f"Black-Litterman optimization failed: {e}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        """Risk parity portfolio optimization"""
        try:
            n_assets = len(returns.columns)
            Sigma = returns.cov() * 252
            
            def risk_budget_objective(weights):
                weights = np.array(weights)
                portfolio_vol = np.sqrt(weights.T @ Sigma.values @ weights)
                marginal_contrib = Sigma.values @ weights / portfolio_vol
                contrib = weights * marginal_contrib
                return np.sum((contrib - portfolio_vol/n_assets)**2)
            
            # Equal risk contribution constraint
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Min 1%, max 50%
            
            # Initial guess (equal weights)
            x0 = np.array([1.0/n_assets] * n_assets)
            
            result = minimize(risk_budget_objective, x0, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = dict(zip(returns.columns, result.x))
                portfolio_return = (returns.mean() * 252) @ result.x
                portfolio_vol = np.sqrt(result.x.T @ Sigma.values @ result.x)
                
                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'expected_volatility': portfolio_vol,
                    'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
                    'optimization_status': 'optimal'
                }
            else:
                return {'optimization_status': 'failed', 'error': result.message}
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {e}")
            return {'optimization_status': 'error', 'error': str(e)}

# Global instance
institutional_optimizer = InstitutionalOptimizer()