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
    
    def _prepare_optimization_data(self, returns_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for optimization - convert to numpy arrays"""
        returns_clean = returns_data.dropna()
        symbols = returns_clean.columns.tolist()
        mu = returns_clean.mean().values * 252  # Annualized returns
        sigma = returns_clean.cov().values * 252  # Annualized covariance
        
        # Add regularization if needed
        if np.linalg.matrix_rank(sigma) < len(symbols):
            self.logger.warning("Singular covariance matrix detected, adding regularization")
            sigma += np.eye(len(symbols)) * 1e-6
        
        return mu, sigma, symbols
    
    def markowitz_optimization(self, returns: pd.DataFrame, 
                         target_return: Optional[float] = None,
                         risk_aversion: float = 1.0) -> Dict:
        """Mean-variance optimization using quadratic programming"""
        try:
            # --- Validate input before optimization ---
            if returns.empty or len(returns.columns) < 2:
                return {
                    'optimization_status': 'failed',
                    'error': 'Not enough assets with valid return history'
                }

            returns = returns.dropna()
            if len(returns) < 30:
                return {
                    'optimization_status': 'failed',
                    'error': f'Insufficient data: only {len(returns)} days available, need 30+'
                }           

            # Prepare data using helper method
            mu, Sigma, symbols = self._prepare_optimization_data(returns)
            n_assets = len(symbols)
        
            # Extra safeguard against invalid values
            if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
                return {
                    'optimization_status': 'failed',
                    'error': 'Covariance matrix contains invalid values'
                }

            # Test matrix positive definiteness
            try:
                np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                self.logger.warning("Non-positive definite covariance, adding regularization")
                Sigma += np.eye(n_assets) * 1e-6
        
            # Define optimization variables
            w = cp.Variable(n_assets)
        
            # Objective functions - FIXED
            portfolio_return = mu @ w  # Simple matrix multiplication
            portfolio_variance = cp.quad_form(w, Sigma)
        
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
        
            if prob.status in ['optimal', 'optimal_inaccurate']:
                # Extract results - FIXED
                weights_array = w.value
                if weights_array is None:
                    return {'optimization_status': 'failed', 'error': 'No solution found'}

                weights = dict(zip(symbols, weights_array))
                expected_return = float(np.dot(mu, weights_array))
                expected_vol = float(np.sqrt(np.dot(weights_array, np.dot(Sigma, weights_array))))

                return {
                    'weights': weights,
                    'expected_return': expected_return,
                    'expected_volatility': expected_vol,
                    'sharpe_ratio': expected_return / expected_vol if expected_vol > 0 else 0,
                    'optimization_status': 'optimal'
                }
            else:
                return {'optimization_status': 'failed', 'error': f'Solver status: {prob.status}'}
            
        except Exception as e:
            self.logger.error(f"Markowitz optimization failed: {e}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def black_litterman_optimization(self, returns: pd.DataFrame,
                                   market_caps: Dict[str, float],
                                   views: Dict[str, float],
                                   tau: float = 0.025) -> Dict:
        """Black-Litterman model with investor views"""
        try:
            if returns.empty or len(returns) < 30:
                return {
                    'optimization_status': 'failed',
                    'error': 'Insufficient data for Black-Litterman optimization'
                }
            
            # Prepare data
            mu, Sigma, symbols = self._prepare_optimization_data(returns)
            n_assets = len(symbols)
            
            # Market capitalization weights
            total_mcap = sum(market_caps.values())
            w_market = np.array([market_caps.get(s, total_mcap/n_assets)/total_mcap for s in symbols])
            
            # Implied equilibrium returns
            risk_aversion = 3.0  # Typical institutional value
            pi = risk_aversion * Sigma @ w_market
            
            # Incorporate views
            if views:
                # Create view matrices
                view_symbols = [s for s in views.keys() if s in symbols]
                n_views = len(view_symbols)
                
                if n_views == 0:
                    self.logger.warning("No valid views found, using market equilibrium")
                    mu_bl = pi
                else:
                    P = np.zeros((n_views, n_assets))
                    Q = np.zeros(n_views)
                    
                    for i, symbol in enumerate(view_symbols):
                        symbol_idx = symbols.index(symbol)
                        P[i, symbol_idx] = 1.0
                        Q[i] = views[symbol]
                    
                    # View uncertainty matrix (simplified approach)
                    Omega = np.eye(n_views) * 0.01
                    
                    try:
                        # Black-Litterman formula
                        M1 = np.linalg.inv(tau * Sigma)
                        M2 = P.T @ np.linalg.inv(Omega) @ P
                        M3 = M1 @ pi
                        M4 = P.T @ np.linalg.inv(Omega) @ Q
                        
                        # New expected returns
                        mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
                    except np.linalg.LinAlgError:
                        self.logger.error("Black-Litterman matrix inversion failed")
                        mu_bl = pi
            else:
                mu_bl = pi
            
            # Create synthetic returns data with Black-Litterman expected returns
            synthetic_returns = pd.DataFrame(index=returns.index[-min(60, len(returns)):])  # Use last 60 days or less
            original_vol = returns.std()
            
            for i, symbol in enumerate(symbols):
                # Generate synthetic returns with BL expected return and historical volatility
                n_periods = len(synthetic_returns)
                daily_expected_return = mu_bl[i] / 252
                daily_vol = original_vol.iloc[i]
                
                synthetic_returns[symbol] = np.random.RandomState(42).normal(
                    daily_expected_return, daily_vol, n_periods
                )
            
            # Run Markowitz optimization on synthetic data
            return self.markowitz_optimization(synthetic_returns)
            
        except Exception as e:
            self.logger.error(f"Black-Litterman optimization failed: {e}")
            return {'optimization_status': 'error', 'error': str(e)}
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        """Risk parity portfolio optimization"""
        try:
            if returns.empty or len(returns) < 30:
                return {
                    'optimization_status': 'failed',
                    'error': 'Insufficient data for risk parity optimization'
                }
            
            # Prepare data
            mu, Sigma, symbols = self._prepare_optimization_data(returns)
            n_assets = len(symbols)
            
            def risk_budget_objective(weights):
                """Minimize sum of squared differences in risk contributions"""
                weights = np.array(weights).flatten()  # Ensure 1D array
                
                # Portfolio volatility
                portfolio_var = weights.T @ Sigma @ weights
                if portfolio_var <= 0:
                    return 1e10  # Large penalty for invalid portfolios
                
                portfolio_vol = np.sqrt(portfolio_var)
                
                # Marginal risk contributions
                marginal_contrib = Sigma @ weights / portfolio_vol
                
                # Risk contributions (weight * marginal contribution)
                risk_contrib = weights * marginal_contrib
                
                # Target: equal risk contribution = portfolio_vol / n_assets
                target_contrib = portfolio_vol / n_assets
                
                # Minimize squared deviations from target
                return np.sum((risk_contrib - target_contrib)**2)
            
            # Constraints: weights sum to 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            
            # Bounds: minimum 1%, maximum 40% per asset
            bounds = [(0.01, 0.40) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            x0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_budget_objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )
            
            if result.success:
                weights = dict(zip(symbols, result.x))
                portfolio_return = float(np.dot(mu, result.x))
                portfolio_vol = float(np.sqrt(result.x.T @ Sigma @ result.x))
                
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