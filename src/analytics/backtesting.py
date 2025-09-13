# src/analytics/backtesting.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime, timedelta

class ProfessionalBacktester:
    """Institutional-grade backtesting with walk-forward analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transaction_costs = 0.001  # 10 bps default
        
    def walk_forward_analysis(self, returns_data: pd.DataFrame,
                            optimization_func: Callable,
                            window_size: int = 252,
                            rebalance_freq: int = 63) -> Dict:
        """Walk-forward optimization analysis"""
        try:
            results = {
                'periods': [],
                'weights_history': [],
                'returns': [],
                'metrics': {}
            }
            
            start_idx = window_size
            end_idx = len(returns_data)
            
            portfolio_returns = []
            
            for current_idx in range(start_idx, end_idx, rebalance_freq):
                # Training window
                train_end = current_idx
                train_start = max(0, train_end - window_size)
                train_data = returns_data.iloc[train_start:train_end]
                
                # Out-of-sample period
                test_start = current_idx
                test_end = min(len(returns_data), current_idx + rebalance_freq)
                test_data = returns_data.iloc[test_start:test_end]
                
                if len(train_data) < 50 or len(test_data) == 0:
                    continue
                
                try:
                    # Optimize portfolio on training data
                    optimization_result = optimization_func(train_data)
                    
                    if 'weights' not in optimization_result:
                        continue
                    
                    weights = optimization_result['weights']
                    
                    # Calculate out-of-sample returns
                    period_returns = self._calculate_portfolio_returns(
                        test_data, weights, include_costs=True
                    )
                    
                    results['periods'].append({
                        'start_date': test_data.index[0],
                        'end_date': test_data.index[-1],
                        'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                        'test_period': f"{test_data.index[0]} to {test_data.index[-1]}"
                    })
                    
                    results['weights_history'].append(weights)
                    results['returns'].append(period_returns)
                    portfolio_returns.extend(period_returns)
                    
                except Exception as e:
                    self.logger.warning(f"Walk-forward period failed: {e}")
                    continue
            
            if not portfolio_returns:
                return {'error': 'No valid walk-forward periods'}
            
            # Aggregate metrics
            portfolio_returns_series = pd.Series(portfolio_returns)
            results['metrics'] = self._calculate_comprehensive_metrics(portfolio_returns_series)
            results['total_periods'] = len(results['periods'])
            results['success_rate'] = len(results['periods']) / ((end_idx - start_idx) // rebalance_freq)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {e}")
            return {'error': str(e)}
    
    def out_of_sample_testing(self, returns_data: pd.DataFrame,
                            optimization_func: Callable,
                            split_date: Optional[str] = None) -> Dict:
        """Out-of-sample backtesting"""
        try:
            # Default split at 70% of data
            if split_date is None:
                split_idx = int(len(returns_data) * 0.7)
                split_date = returns_data.index[split_idx]
            
            # Split data
            in_sample = returns_data[:split_date]
            out_sample = returns_data[split_date:]
            
            if len(in_sample) < 100 or len(out_sample) < 50:
                return {'error': 'Insufficient data for in/out-of-sample testing'}
            
            # Optimize on in-sample data
            optimization_result = optimization_func(in_sample)
            
            if 'weights' not in optimization_result:
                return {'error': 'Optimization failed'}
            
            weights = optimization_result['weights']
            
            # Test on out-of-sample data
            in_sample_returns = self._calculate_portfolio_returns(in_sample, weights)
            out_sample_returns = self._calculate_portfolio_returns(out_sample, weights, include_costs=True)
            
            # Calculate metrics for both periods
            in_sample_metrics = self._calculate_comprehensive_metrics(pd.Series(in_sample_returns))
            out_sample_metrics = self._calculate_comprehensive_metrics(pd.Series(out_sample_returns))
            
            return {
                'split_date': split_date,
                'in_sample_period': f"{in_sample.index[0]} to {in_sample.index[-1]}",
                'out_sample_period': f"{out_sample.index[0]} to {out_sample.index[-1]}",
                'optimal_weights': weights,
                'in_sample_metrics': in_sample_metrics,
                'out_sample_metrics': out_sample_metrics,
                'performance_degradation': {
                    'return_diff': out_sample_metrics['annualized_return'] - in_sample_metrics['annualized_return'],
                    'sharpe_diff': out_sample_metrics['sharpe_ratio'] - in_sample_metrics['sharpe_ratio'],
                    'vol_diff': out_sample_metrics['volatility'] - in_sample_metrics['volatility']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Out-of-sample testing failed: {e}")
            return {'error': str(e)}
    
    def regime_analysis(self, returns_data: pd.DataFrame,
                      strategy_func: Callable,
                      regime_indicator: pd.Series) -> Dict:
        """Performance analysis across different market regimes"""
        try:
            regimes = regime_indicator.unique()
            regime_results = {}
            
            for regime in regimes:
                regime_mask = regime_indicator == regime
                regime_data = returns_data[regime_mask]
                
                if len(regime_data) < 20:
                    continue
                
                # Apply strategy to regime data
                strategy_result = strategy_func(regime_data)
                
                if 'weights' not in strategy_result:
                    continue
                
                weights = strategy_result['weights']
                regime_returns = self._calculate_portfolio_returns(regime_data, weights)
                regime_metrics = self._calculate_comprehensive_metrics(pd.Series(regime_returns))
                
                regime_results[regime] = {
                    'period_count': regime_mask.sum(),
                    'frequency': regime_mask.mean(),
                    'metrics': regime_metrics,
                    'optimal_weights': weights
                }
            
            return {
                'regime_results': regime_results,
                'best_regime': max(regime_results.keys(), 
                                 key=lambda x: regime_results[x]['metrics']['sharpe_ratio']),
                'worst_regime': min(regime_results.keys(), 
                                  key=lambda x: regime_results[x]['metrics']['sharpe_ratio']),
                'regime_consistency': self._calculate_regime_consistency(regime_results)
            }
            
        except Exception as e:
            self.logger.error(f"Regime analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame,
                                   weights: Dict[str, float],
                                   include_costs: bool = False) -> List[float]:
        """Calculate portfolio returns with optional transaction costs"""
        portfolio_returns = []
        
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        for idx, date in enumerate(returns_data.index):
            daily_return = 0.0
            
            for symbol, weight in weights.items():
                if symbol in returns_data.columns:
                    asset_return = returns_data.loc[date, symbol]
                    if not pd.isna(asset_return):
                        daily_return += weight * asset_return
            
            # Apply transaction costs (simplified)
            if include_costs and idx > 0:
                daily_return -= self.transaction_costs / 252  # Daily cost
            
            portfolio_returns.append(daily_return)
        
        return portfolio_returns
    
    def _calculate_comprehensive_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0 or returns.isna().all():
            return {'error': 'No valid returns data'}
        
        returns = returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win * (returns > 0).sum() / (avg_loss * (returns < 0).sum())) if avg_loss != 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() if (returns <= np.percentile(returns, 5)).any() else 0
        }
    
    def _calculate_regime_consistency(self, regime_results: Dict) -> float:
        """Calculate consistency of strategy across regimes"""
        if len(regime_results) < 2:
            return 1.0
        
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in regime_results.values()]
        return 1.0 - (np.std(sharpe_ratios) / np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else 0

# Global instance
professional_backtester = ProfessionalBacktester()