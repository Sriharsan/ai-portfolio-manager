# src/models/institutional_risk.py

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from typing import Dict, List, Optional, Tuple
import logging

class InstitutionalRiskManager:
    """Enterprise-grade risk management with GARCH, Monte Carlo, and stress testing"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.logger = logging.getLogger(__name__)
    
    def monte_carlo_var(self, returns: pd.Series, 
                       portfolio_value: float = 1000000,
                       simulations: int = 10000,
                       horizon: int = 1) -> Dict:
        """Monte Carlo VaR simulation"""
        try:
            # Fit distribution to returns
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            random_returns = np.random.normal(mu, sigma, (simulations, horizon))
            
            # Calculate portfolio values
            scenario_returns = np.sum(random_returns, axis=1) if horizon > 1 else random_returns.flatten()
            scenario_values = portfolio_value * (1 + scenario_returns)
            portfolio_changes = scenario_values - portfolio_value
            
            results = {}
            for confidence in self.confidence_levels:
                var = np.percentile(portfolio_changes, (1 - confidence) * 100)
                cvar = portfolio_changes[portfolio_changes <= var].mean()
                
                results[f'var_{int(confidence*100)}'] = var
                results[f'cvar_{int(confidence*100)}'] = cvar
            
            results.update({
                'method': 'monte_carlo',
                'simulations': simulations,
                'horizon_days': horizon,
                'mean_scenario_return': np.mean(scenario_returns),
                'worst_case_return': np.min(scenario_returns),
                'best_case_return': np.max(scenario_returns)
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Monte Carlo VaR failed: {e}")
            return {'error': str(e)}
    
    def garch_volatility_forecast(self, returns: pd.Series, 
                                 horizon: int = 22) -> Dict:
        """GARCH volatility forecasting"""
        try:
            # Remove NaN and scale returns to percentage
            clean_returns = returns.dropna() * 100
            
            if len(clean_returns) < 100:
                return {'error': 'Insufficient data for GARCH modeling'}
            
            # Fit GARCH(1,1) model
            model = arch_model(clean_returns, vol='GARCH', p=1, q=1, dist='normal')
            fitted_model = model.fit(disp='off')
            
            # Forecast volatility
            forecasts = fitted_model.forecast(horizon=horizon, method='simulation')
            forecast_vol = np.sqrt(forecasts.variance.values[-1, :]) / 100  # Convert back
            
            # Calculate volatility statistics
            current_vol = clean_returns.std() / 100 * np.sqrt(252)
            forecast_vol_annual = forecast_vol * np.sqrt(252)
            
            return {
                'current_volatility': current_vol,
                'forecast_volatility': forecast_vol_annual.tolist(),
                'mean_forecast_vol': np.mean(forecast_vol_annual),
                'vol_trend': 'increasing' if np.mean(forecast_vol_annual) > current_vol else 'decreasing',
                'model_aic': fitted_model.aic,
                'model_bic': fitted_model.bic,
                'horizon_days': horizon
            }
            
        except Exception as e:
            self.logger.error(f"GARCH forecasting failed: {e}")
            # Fallback to simple volatility
            vol = returns.std() * np.sqrt(252)
            return {
                'current_volatility': vol,
                'forecast_volatility': [vol] * horizon,
                'mean_forecast_vol': vol,
                'vol_trend': 'stable',
                'method': 'fallback'
            }
    
    def stress_test_scenarios(self, portfolio_weights: Dict[str, float],
                            returns_data: pd.DataFrame,
                            portfolio_value: float = 1000000) -> Dict:
        """Comprehensive stress testing scenarios"""
        try:
            scenarios = {
                '2008_financial_crisis': {'equity_shock': -0.35, 'bond_rally': 0.15, 'vol_spike': 3.0},
                '2020_covid_crash': {'equity_shock': -0.30, 'bond_rally': 0.08, 'vol_spike': 4.0},
                'interest_rate_shock': {'bond_shock': -0.20, 'equity_impact': -0.15, 'vol_increase': 2.0},
                'inflation_surge': {'equity_shock': -0.10, 'bond_shock': -0.15, 'real_assets': 0.20},
                'geopolitical_crisis': {'equity_shock': -0.25, 'safe_haven': 0.10, 'vol_spike': 2.5}
            }
            
            stress_results = {}
            
            for scenario_name, shocks in scenarios.items():
                portfolio_impact = 0.0
                
                for symbol, weight in portfolio_weights.items():
                    # Categorize assets and apply shocks
                    if symbol in ['SPY', 'QQQ', 'IWM', 'VEA', 'VWO']:  # Equity
                        impact = shocks.get('equity_shock', 0)
                    elif symbol in ['TLT', 'IEF', 'LQD', 'SHY']:  # Bonds
                        impact = shocks.get('bond_rally', shocks.get('bond_shock', 0))
                    elif symbol in ['GLD', 'SLV', 'USO']:  # Commodities
                        impact = shocks.get('real_assets', shocks.get('safe_haven', 0))
                    else:
                        impact = shocks.get('equity_shock', 0) * 0.5  # Default
                    
                    portfolio_impact += weight * impact
                
                # Calculate scenario portfolio value
                scenario_value = portfolio_value * (1 + portfolio_impact)
                scenario_loss = scenario_value - portfolio_value
                
                stress_results[scenario_name] = {
                    'portfolio_return': portfolio_impact,
                    'portfolio_value': scenario_value,
                    'loss_amount': scenario_loss,
                    'loss_percentage': portfolio_impact,
                    'severity': self._categorize_loss(abs(portfolio_impact))
                }
            
            # Overall stress test summary
            worst_case_loss = min([r['loss_percentage'] for r in stress_results.values()])
            average_loss = np.mean([r['loss_percentage'] for r in stress_results.values()])
            
            return {
                'scenarios': stress_results,
                'worst_case_loss': worst_case_loss,
                'average_stress_loss': average_loss,
                'stress_test_date': pd.Timestamp.now(),
                'portfolio_resilience': self._assess_resilience(abs(worst_case_loss))
            }
            
        except Exception as e:
            self.logger.error(f"Stress testing failed: {e}")
            return {'error': str(e)}
    
    def correlation_breakdown_analysis(self, returns_data: pd.DataFrame) -> Dict:
        """Dynamic correlation analysis with breakdown detection"""
        try:
            # Rolling correlation calculation
            window = 60  # 3-month window
            correlations = {}
            
            symbols = returns_data.columns.tolist()
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    rolling_corr = returns_data[symbol1].rolling(window).corr(returns_data[symbol2])
                    correlations[f"{symbol1}_{symbol2}"] = {
                        'current_correlation': rolling_corr.iloc[-1] if not rolling_corr.empty else 0,
                        'mean_correlation': rolling_corr.mean(),
                        'max_correlation': rolling_corr.max(),
                        'min_correlation': rolling_corr.min(),
                        'volatility': rolling_corr.std(),
                        'trend': self._correlation_trend(rolling_corr)
                    }
            
            # Overall correlation matrix
            current_corr_matrix = returns_data.corr()
            
            # Identify high correlation clusters
            high_corr_pairs = [
                pair for pair, stats in correlations.items() 
                if abs(stats['current_correlation']) > 0.7
            ]
            
            return {
                'pairwise_correlations': correlations,
                'current_correlation_matrix': current_corr_matrix.to_dict(),
                'high_correlation_pairs': high_corr_pairs,
                'mean_portfolio_correlation': np.mean([
                    abs(stats['current_correlation']) 
                    for stats in correlations.values()
                ]),
                'correlation_risk': 'high' if len(high_corr_pairs) > len(symbols) else 'moderate'
            }
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return {'error': str(e)}
    
    def _categorize_loss(self, loss_percentage: float) -> str:
        """Categorize loss severity"""
        if loss_percentage > 0.30:
            return 'severe'
        elif loss_percentage > 0.20:
            return 'high'
        elif loss_percentage > 0.10:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_resilience(self, worst_loss: float) -> str:
        """Assess portfolio resilience"""
        if worst_loss < 0.15:
            return 'strong'
        elif worst_loss < 0.25:
            return 'moderate'
        else:
            return 'weak'
    
    def _correlation_trend(self, correlation_series: pd.Series) -> str:
        """Determine correlation trend"""
        if len(correlation_series) < 10:
            return 'insufficient_data'
        
        recent = correlation_series.tail(10).mean()
        earlier = correlation_series.head(10).mean()
        
        if recent > earlier * 1.1:
            return 'increasing'
        elif recent < earlier * 0.9:
            return 'decreasing'
        else:
            return 'stable'

# Global instance
institutional_risk_manager = InstitutionalRiskManager()