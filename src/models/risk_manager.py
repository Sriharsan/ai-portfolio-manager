# src/models/risk_manager_fixed.py - CORRECTED VERSION WITH PROPER DATA HANDLING

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

# Import configuration and other components with correct paths
try:
    from ..config import config
    from .institutional_risk import institutional_risk_manager
except ImportError:
    try:
        # Alternative import for when running from different contexts
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config
        from models.institutional_risk import institutional_risk_manager
    except ImportError:
        # Fallback for standalone usage
        config = None
        institutional_risk_manager = None

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskProfile:
    """Dynamic risk profile configuration"""
    confidence_levels: List[float]
    max_position_size: float
    max_sector_concentration: float
    target_volatility: float
    stress_test_scenarios: List[str]
    rebalancing_threshold: float
    
    @classmethod
    def from_risk_tolerance(cls, risk_tolerance: str) -> 'RiskProfile':
        """Create risk profile from risk tolerance string"""
        profiles = {
            'conservative': cls(
                confidence_levels=[0.90, 0.95, 0.99],
                max_position_size=0.15,
                max_sector_concentration=0.25,
                target_volatility=0.08,
                stress_test_scenarios=['2008_financial_crisis', 'interest_rate_shock'],
                rebalancing_threshold=0.05
            ),
            'moderate': cls(
                confidence_levels=[0.95, 0.99],
                max_position_size=0.25,
                max_sector_concentration=0.35,
                target_volatility=0.12,
                stress_test_scenarios=['2008_financial_crisis', '2020_covid_crash', 'interest_rate_shock'],
                rebalancing_threshold=0.08
            ),
            'aggressive': cls(
                confidence_levels=[0.95, 0.99],
                max_position_size=0.35,
                max_sector_concentration=0.50,
                target_volatility=0.18,
                stress_test_scenarios=['2008_financial_crisis', '2020_covid_crash', 'geopolitical_crisis'],
                rebalancing_threshold=0.10
            )
        }
        return profiles.get(risk_tolerance, profiles['moderate'])

class DynamicRiskManager:
    """
    FIXED Dynamic Risk Manager with robust data validation
    """
    
    def __init__(self, 
                 risk_profile: Optional[RiskProfile] = None,
                 portfolio_value: float = None,
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize dynamic risk manager with proper config integration
        """
        self.logger = logging.getLogger(__name__)
        
        # Use your config system properly
        if config:
            self.portfolio_value = portfolio_value or 1000000
            self.risk_free_rate = config.DEFAULT_RISK_FREE_RATE
            self.max_position_size = config.MAX_POSITION_SIZE
            self.logger.info("✅ Config system integrated successfully")
        else:
            self.portfolio_value = portfolio_value or 1000000
            self.risk_free_rate = 0.02
            self.max_position_size = 0.25
            self.logger.warning("⚠️ Config system not available, using defaults")
        
        # Set risk profile
        self.risk_profile = risk_profile or RiskProfile.from_risk_tolerance('moderate')
        self.benchmark_symbol = benchmark_symbol
        
        # Connect to your institutional risk manager
        self.institutional_risk = institutional_risk_manager
        if self.institutional_risk:
            self.logger.info("✅ Connected to institutional risk manager")
        else:
            self.logger.warning("⚠️ Institutional risk manager not available")
    
    def validate_portfolio_data(self, data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """
        CRITICAL: Validate portfolio data before any risk calculations
        """
        validation_result = {
            'is_valid': False,
            'error_message': None,
            'data_shape': None,
            'date_range': None,
            'missing_values': 0,
            'min_observations': 30
        }
        
        try:
            if data is None:
                validation_result['error_message'] = 'No data provided'
                return validation_result
            
            if isinstance(data, pd.DataFrame):
                data_clean = data.dropna()
                validation_result['data_shape'] = data_clean.shape
                validation_result['missing_values'] = data.isnull().sum().sum()
            else:  # Series
                data_clean = data.dropna()
                validation_result['data_shape'] = (len(data_clean), 1)
                validation_result['missing_values'] = data.isnull().sum()
            
            if len(data_clean) < validation_result['min_observations']:
                validation_result['error_message'] = f'Insufficient data: {len(data_clean)} observations, need {validation_result["min_observations"]}+'
                return validation_result
            
            if isinstance(data_clean.index, pd.DatetimeIndex):
                validation_result['date_range'] = (data_clean.index.min(), data_clean.index.max())
            
            # Check for valid numerical data
            if isinstance(data_clean, pd.DataFrame):
                if not data_clean.select_dtypes(include=[np.number]).shape[1]:
                    validation_result['error_message'] = 'No numerical columns found'
                    return validation_result
            else:
                if not pd.api.types.is_numeric_dtype(data_clean):
                    validation_result['error_message'] = 'Non-numerical data provided'
                    return validation_result
            
            validation_result['is_valid'] = True
            self.logger.info(f"✅ Data validation passed: {validation_result['data_shape']} shape, {validation_result['date_range']}")
            
        except Exception as e:
            validation_result['error_message'] = f'Data validation error: {str(e)}'
            self.logger.error(validation_result['error_message'])
        
        return validation_result
    
    def calculate_portfolio_returns_from_weights(self, 
                                               price_data: pd.DataFrame, 
                                               weights: Dict[str, float]) -> pd.Series:
        """
        FIXED: Calculate portfolio returns from price data and weights
        """
        try:
            # Validate inputs
            if not price_data.empty and weights:
                # Calculate returns
                returns_data = price_data.pct_change().dropna()
                
                # Filter weights to match available assets
                available_assets = set(returns_data.columns) & set(weights.keys())
                if not available_assets:
                    self.logger.error("No matching assets between weights and price data")
                    return pd.Series(dtype=float)
                
                # Normalize weights for available assets
                total_weight = sum(weights[asset] for asset in available_assets)
                if total_weight <= 0:
                    self.logger.error("Total weight is zero or negative")
                    return pd.Series(dtype=float)
                
                normalized_weights = {
                    asset: weights[asset] / total_weight 
                    for asset in available_assets
                }
                
                # Calculate weighted portfolio returns
                portfolio_returns = pd.Series(0.0, index=returns_data.index)
                for asset, weight in normalized_weights.items():
                    if asset in returns_data.columns:
                        portfolio_returns += returns_data[asset] * weight
                
                self.logger.info(f"✅ Portfolio returns calculated: {len(portfolio_returns)} observations")
                return portfolio_returns
            
            else:
                self.logger.error("Empty price data or weights")
                return pd.Series(dtype=float)
                
        except Exception as e:
            self.logger.error(f"Portfolio returns calculation failed: {e}")
            return pd.Series(dtype=float)
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     method: str = 'historical',
                     confidence_level: Optional[float] = None,
                     portfolio_value: Optional[float] = None) -> Dict[str, float]:
        """
        FIXED VaR calculation with proper data validation
        """
        # Use provided portfolio value or instance default
        portfolio_val = portfolio_value or self.portfolio_value
        
        # Validate data first
        validation = self.validate_portfolio_data(returns)
        if not validation['is_valid']:
            self.logger.error(f"VaR calculation failed validation: {validation['error_message']}")
            return {
                'error': validation['error_message'],
                'method': 'validation_failed',
                'portfolio_value': portfolio_val
            }
        
        results = {}
        confidence_levels = [confidence_level] if confidence_level else self.risk_profile.confidence_levels
        
        try:
            # Clean data
            clean_returns = returns.dropna()
            if len(clean_returns) == 0:
                return {'error': 'All return values are NaN', 'portfolio_value': portfolio_val}
            
            for conf_level in confidence_levels:
                conf_pct = int(conf_level * 100)
                
                if method == 'historical':
                    var_return = np.percentile(clean_returns, (1 - conf_level) * 100)
                    
                elif method == 'parametric':
                    mu = clean_returns.mean()
                    sigma = clean_returns.std()
                    if sigma <= 0:
                        var_return = 0.0
                    else:
                        var_return = stats.norm.ppf(1 - conf_level, mu, sigma)
                    
                elif method == 'monte_carlo' and self.institutional_risk:
                    # Use your institutional risk manager's Monte Carlo
                    mc_result = self.institutional_risk.monte_carlo_var(
                        clean_returns, portfolio_val, simulations=5000
                    )
                    if f'var_{conf_pct}' in mc_result:
                        var_return = mc_result[f'var_{conf_pct}'] / portfolio_val
                    else:
                        var_return = np.percentile(clean_returns, (1 - conf_level) * 100)
                else:
                    # Fallback to historical
                    var_return = np.percentile(clean_returns, (1 - conf_level) * 100)
                
                # Convert to dollar terms - FIXED
                var_dollar = abs(var_return * portfolio_val)  # Take absolute value for clarity
                
                results[f'var_{conf_pct}_return'] = var_return
                results[f'var_{conf_pct}_dollar'] = var_dollar
        
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return {'error': str(e), 'portfolio_value': portfolio_val}
        
        results.update({
            'method': method,
            'observation_count': len(clean_returns),
            'portfolio_value': portfolio_val,
            'data_range': f"{clean_returns.index.min()} to {clean_returns.index.max()}" if hasattr(clean_returns.index, 'min') else 'Unknown'
        })
        
        self.logger.info(f"✅ VaR calculated successfully: {results}")
        return results
    
    def calculate_cvar(self, 
                      returns: pd.Series, 
                      confidence_level: Optional[float] = None,
                      portfolio_value: Optional[float] = None) -> Dict:
        """
        FIXED CVaR calculation with proper data validation
        """
        portfolio_val = portfolio_value or self.portfolio_value
        
        # Validate data
        validation = self.validate_portfolio_data(returns)
        if not validation['is_valid']:
            return {
                'error': validation['error_message'],
                'portfolio_value': portfolio_val
            }
        
        results = {}
        confidence_levels = [confidence_level] if confidence_level else self.risk_profile.confidence_levels
        
        try:
            clean_returns = returns.dropna()
            
            for conf_level in confidence_levels:
                conf_pct = int(conf_level * 100)
                
                var_return = np.percentile(clean_returns, (1 - conf_level) * 100)
                tail_returns = clean_returns[clean_returns <= var_return]
                
                if len(tail_returns) > 0:
                    cvar_return = tail_returns.mean()
                else:
                    cvar_return = var_return
                
                cvar_dollar = abs(cvar_return * portfolio_val)
                
                results[f'cvar_{conf_pct}_return'] = cvar_return
                results[f'cvar_{conf_pct}_dollar'] = cvar_dollar
        
        except Exception as e:
            self.logger.error(f"CVaR calculation failed: {e}")
            return {'error': str(e), 'portfolio_value': portfolio_val}
        
        results['portfolio_value'] = portfolio_val
        results['observation_count'] = len(clean_returns)
        return results
    
    def max_drawdown(self, returns: pd.Series) -> float:
        """
        FIXED Maximum drawdown calculation
        """
        try:
            validation = self.validate_portfolio_data(returns)
            if not validation['is_valid']:
                self.logger.warning(f"Max drawdown validation failed: {validation['error_message']}")
                return 0.0
            
            clean_returns = returns.dropna()
            if len(clean_returns) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cumulative_returns = (1 + clean_returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Return maximum drawdown (most negative value)
            max_dd = float(drawdowns.min())
            
            self.logger.info(f"✅ Max drawdown calculated: {max_dd:.4f}")
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0
    
    def comprehensive_portfolio_analysis(self, 
                                       returns: pd.Series,
                                       portfolio_value: Optional[float] = None,
                                       benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        FIXED Comprehensive analysis with robust data validation
        """
        portfolio_val = portfolio_value or self.portfolio_value
        
        # Validate input data
        validation = self.validate_portfolio_data(returns)
        if not validation['is_valid']:
            return {
                'error': f'Data validation failed: {validation["error_message"]}',
                'validation_details': validation,
                'portfolio_value': portfolio_val
            }
        
        try:
            results = {
                'portfolio_value': portfolio_val,
                'data_validation': validation
            }
            
            # 1. Basic risk metrics
            results['basic_metrics'] = self._calculate_basic_metrics(returns)
            
            # 2. VaR analysis (multiple methods)
            results['var_analysis'] = self.calculate_var(returns, method='historical', portfolio_value=portfolio_val)
            
            # 3. CVaR analysis  
            results['cvar_analysis'] = self.calculate_cvar(returns, portfolio_value=portfolio_val)
            
            # 4. Beta analysis (if benchmark provided)
            if benchmark_returns is not None:
                results['beta_analysis'] = self.calculate_beta(returns, benchmark_returns)
            
            # 5. Advanced institutional analysis (if available)
            if self.institutional_risk and len(returns.dropna()) >= 50:
                # Monte Carlo VaR
                mc_result = self.institutional_risk.monte_carlo_var(
                    returns, portfolio_val, simulations=10000
                )
                if 'error' not in mc_result:
                    results['monte_carlo'] = mc_result
                
                # GARCH volatility forecast
                garch_result = self.institutional_risk.garch_volatility_forecast(returns)
                if 'error' not in garch_result:
                    results['volatility_forecast'] = garch_result
            
            # 6. Risk assessment and compliance
            results['risk_level'] = self._assess_risk_level(results)
            results['compliance_status'] = self._check_compliance(results)
            
            # 7. Recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            results['analysis_timestamp'] = pd.Timestamp.now()
            
            self.logger.info("✅ Comprehensive portfolio analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'error': str(e),
                'portfolio_value': portfolio_val,
                'validation_details': validation
            }
    
    def calculate_beta(self, 
                      portfolio_returns: pd.Series, 
                      benchmark_returns: pd.Series) -> Dict:
        """
        FIXED BETA CALCULATION with proper alignment
        """
        try:
            # Validate both series
            port_validation = self.validate_portfolio_data(portfolio_returns)
            bench_validation = self.validate_portfolio_data(benchmark_returns)
            
            if not port_validation['is_valid']:
                return {'error': f'Portfolio data invalid: {port_validation["error_message"]}', 'beta_fallback': 1.0}
            
            if not bench_validation['is_valid']:
                return {'error': f'Benchmark data invalid: {bench_validation["error_message"]}', 'beta_fallback': 1.0}
            
            # Align the series properly
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
            aligned_data.columns = ['portfolio', 'benchmark'] 
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 30:
                return {
                    'error': 'Insufficient aligned data for beta calculation',
                    'beta_fallback': 1.0,
                    'observations': len(aligned_data)
                }
            
            # Proper covariance-based beta calculation
            portfolio_rets = aligned_data['portfolio'].values
            benchmark_rets = aligned_data['benchmark'].values
            
            # Beta = Covariance(portfolio, benchmark) / Variance(benchmark)
            covariance = np.cov(portfolio_rets, benchmark_rets)[0, 1]
            benchmark_variance = np.var(benchmark_rets, ddof=1)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            else:
                beta = 1.0
            
            # Calculate correlation and alpha
            correlation = np.corrcoef(portfolio_rets, benchmark_rets)[0, 1]
            
            # Alpha calculation (annualized)
            portfolio_mean_annual = np.mean(portfolio_rets) * 252
            benchmark_mean_annual = np.mean(benchmark_rets) * 252
            alpha = portfolio_mean_annual - (self.risk_free_rate + beta * (benchmark_mean_annual - self.risk_free_rate))
            
            result = {
                'beta': beta,
                'alpha': alpha,
                'correlation': correlation,
                'observations': len(aligned_data),
                'benchmark': self.benchmark_symbol,
                'r_squared': correlation ** 2,
                'tracking_error': np.std(portfolio_rets - beta * benchmark_rets) * np.sqrt(252)
            }
            
            self.logger.info(f"✅ Beta analysis completed: beta={beta:.3f}, alpha={alpha:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Beta calculation failed: {e}")
            return {
                'error': str(e),
                'beta_fallback': 1.0,
                'alpha_fallback': 0.0
            }
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict:
        """FIXED basic metrics calculation"""
        try:
            clean_returns = returns.dropna()
            if len(clean_returns) == 0:
                return {'error': 'No valid return data'}
            
            metrics = {}
            
            # Volatility (annualized)
            daily_vol = clean_returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            metrics['volatility_daily'] = daily_vol
            metrics['volatility_annual'] = annual_vol
            
            # Downside deviation
            downside_returns = clean_returns[clean_returns < 0]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(252)
            else:
                downside_vol = 0.0
            metrics['downside_deviation'] = downside_vol
            
            # Maximum drawdown using fixed method
            metrics['max_drawdown'] = self.max_drawdown(clean_returns)
            
            # Sharpe ratio
            mean_return = clean_returns.mean()
            excess_returns = mean_return - (self.risk_free_rate / 252)
            if daily_vol > 0:
                sharpe_ratio = (excess_returns / daily_vol) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Sortino ratio
            if downside_vol > 0:
                sortino_ratio = (excess_returns * np.sqrt(252)) / downside_vol
            else:
                sortino_ratio = 0.0
            metrics['sortino_ratio'] = sortino_ratio
            
            # Additional metrics
            metrics['annualized_return'] = mean_return * 252
            metrics['skewness'] = clean_returns.skew()
            metrics['kurtosis'] = clean_returns.kurtosis()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Basic metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _assess_risk_level(self, risk_metrics: Dict) -> str:
        """Assess overall risk level"""
        risk_indicators = []
        
        if 'basic_metrics' in risk_metrics and 'error' not in risk_metrics['basic_metrics']:
            metrics = risk_metrics['basic_metrics']
            
            # Volatility check
            vol = metrics.get('volatility_annual', 0)
            if vol > 0.25:
                risk_indicators.append('high_volatility')
            elif vol > 0.15:
                risk_indicators.append('moderate_volatility')
            
            # Drawdown check
            dd = abs(metrics.get('max_drawdown', 0))
            if dd > 0.30:
                risk_indicators.append('high_drawdown')
            elif dd > 0.15:
                risk_indicators.append('moderate_drawdown')
        
        # VaR check
        if 'var_analysis' in risk_metrics and 'var_95_return' in risk_metrics.get('var_analysis', {}):
            var_95 = abs(risk_metrics['var_analysis']['var_95_return'])
            if var_95 > 0.05:
                risk_indicators.append('high_var')
        
        # Risk level determination
        high_risk_count = sum(1 for indicator in risk_indicators if 'high' in indicator)
        
        if high_risk_count >= 2:
            return RiskLevel.HIGH.value
        elif high_risk_count >= 1 or len(risk_indicators) >= 3:
            return RiskLevel.MODERATE.value
        else:
            return RiskLevel.LOW.value
    
    def _check_compliance(self, risk_metrics: Dict) -> Dict:
        """Check compliance with risk profile"""
        compliance = {}
        
        if 'basic_metrics' in risk_metrics and 'error' not in risk_metrics['basic_metrics']:
            metrics = risk_metrics['basic_metrics']
            
            # Volatility compliance
            vol = metrics.get('volatility_annual', 0)
            compliance['volatility_compliant'] = vol <= (self.risk_profile.target_volatility * 1.2)
            
            # Drawdown compliance
            dd = abs(metrics.get('max_drawdown', 0))
            compliance['drawdown_compliant'] = dd <= (self.risk_profile.target_volatility * 2)
        else:
            compliance['volatility_compliant'] = False
            compliance['drawdown_compliant'] = False
        
        compliance['overall_compliant'] = all(compliance.values()) if compliance else False
        return compliance
    
    def _generate_recommendations(self, risk_metrics: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Check for data issues first
        if 'error' in risk_metrics:
            recommendations.append("⚠️ Data quality issues detected - ensure sufficient historical data")
            recommendations.append("📊 Minimum 30 trading days of return data required for reliable risk metrics")
            return recommendations
        
        risk_level = risk_metrics.get('risk_level', 'moderate')
        
        if risk_level == 'high':
            recommendations.append("🔴 HIGH RISK: Consider reducing position sizes immediately")
            recommendations.append("🔄 Increase diversification across asset classes")
            recommendations.append("📊 Review correlation structure for concentration risk")
        elif risk_level == 'moderate':
            recommendations.append("🟡 MODERATE RISK: Monitor risk metrics closely")
            recommendations.append("⚖️ Consider rebalancing if deviations exceed thresholds")
        else:
            recommendations.append("🟢 LOW RISK: Current risk level is acceptable")
            recommendations.append("📈 Consider opportunities for enhanced returns")
        
        # Compliance recommendations
        compliance = risk_metrics.get('compliance_status', {})
        if not compliance.get('overall_compliant', True):
            recommendations.append("⚠️ Portfolio exceeds risk limits - immediate rebalancing recommended")
        
        return recommendations

# =============================================================================
# FACTORY FUNCTIONS - CORRECT WAY TO CREATE INSTANCES
# =============================================================================

def create_risk_manager(risk_tolerance: str = 'moderate', 
                       portfolio_value: float = 1000000,
                       benchmark: str = 'SPY') -> DynamicRiskManager:
    """
    Factory function to create properly configured risk manager
    THIS IS THE CORRECT WAY TO INSTANTIATE
    """
    risk_profile = RiskProfile.from_risk_tolerance(risk_tolerance)
    return DynamicRiskManager(
        risk_profile=risk_profile, 
        portfolio_value=portfolio_value,
        benchmark_symbol=benchmark
    )

# =============================================================================
# DEBUGGING FUNCTIONS
# =============================================================================

def debug_risk_calculation(returns_data, weights, portfolio_value=1000000):
    """
    Debug function to identify risk calculation issues
    """
    print("🔍 DEBUG: Risk Calculation Analysis")
    print("=" * 50)
    
    # Create risk manager
    risk_mgr = create_risk_manager(portfolio_value=portfolio_value)
    
    # Debug data
    if isinstance(returns_data, pd.DataFrame):
        portfolio_returns = risk_mgr.calculate_portfolio_returns_from_weights(
            returns_data, weights
        )
        print(f"📊 Portfolio returns calculated: {len(portfolio_returns)} observations")
        print(f"📈 Return stats: mean={portfolio_returns.mean():.6f}, std={portfolio_returns.std():.6f}")
        
        # Run validation
        validation = risk_mgr.validate_portfolio_data(portfolio_returns)
        print(f"✅ Data validation: {validation}")
        
        # Calculate VaR
        var_result = risk_mgr.calculate_var(portfolio_returns, portfolio_value=portfolio_value)
        print(f"💰 VaR Result: {var_result}")
        
        # Calculate max drawdown
        max_dd = risk_mgr.max_drawdown(portfolio_returns)
        print(f"📉 Max Drawdown: {max_dd:.4f}")
        
        return {
            'portfolio_returns': portfolio_returns,
            'validation': validation,
            'var_result': var_result,
            'max_drawdown': max_dd
        }
    
    else:
        print("❌ Invalid returns data provided")
        return None

# =============================================================================
# GLOBAL INSTANCE - Now properly configured
# =============================================================================
risk_manager = create_risk_manager()  # This is now DYNAMIC, not static!