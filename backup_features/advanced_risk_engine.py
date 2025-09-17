import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
from enum import Enum

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"

class StressScenario(Enum):
    """Predefined stress test scenarios."""
    MARKET_CRASH_2008 = "2008_financial_crisis"
    COVID_PANDEMIC = "covid_pandemic_2020"
    DOT_COM_BUBBLE = "dot_com_bubble_2000"
    CUSTOM = "custom_scenario"

@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: datetime
    risk_type: str
    severity: RiskLevel
    message: str
    portfolio_id: str
    metric_value: float
    threshold: float
    action_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StressTestResult:
    """Stress test results."""
    scenario: StressScenario
    portfolio_value_change: float
    individual_asset_impacts: Dict[str, float]
    risk_metrics_under_stress: Dict[str, float]
    probability_of_scenario: float
    recovery_time_estimate: int  # days
    recommended_actions: List[str]

class EnterpriseRiskEngine:
    """
    Advanced risk management engine for institutional portfolios.
    Implements comprehensive risk monitoring and stress testing.
    """
    
    def __init__(self):
        self.risk_alerts = []
        self.stress_scenarios = self._initialize_stress_scenarios()
        self.risk_thresholds = self._default_risk_thresholds()
        self.monitoring_active = False
    
    def _initialize_stress_scenarios(self) -> Dict[StressScenario, Dict]:
        """Initialize predefined stress test scenarios."""
        return {
            StressScenario.MARKET_CRASH_2008: {
                'equity_shock': -0.37,  # S&P 500 peak-to-trough
                'bond_shock': 0.05,     # Flight to quality
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.8,
                'duration_days': 180
            },
            StressScenario.COVID_PANDEMIC: {
                'equity_shock': -0.34,  # March 2020 crash
                'bond_shock': 0.08,     # Government bonds rallied
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.9,
                'duration_days': 30
            },
            StressScenario.DOT_COM_BUBBLE: {
                'equity_shock': -0.49,  # NASDAQ crash
                'bond_shock': 0.12,     # Strong flight to quality
                'volatility_multiplier': 2.2,
                'correlation_increase': 0.7,
                'duration_days': 365
            }
        }
    
    def _default_risk_thresholds(self) -> Dict[str, Dict]:
        """Default risk monitoring thresholds."""
        return {
            'var_95': {'moderate': 0.05, 'high': 0.08, 'critical': 0.12},
            'max_drawdown': {'moderate': 0.10, 'high': 0.15, 'critical': 0.25},
            'concentration_risk': {'moderate': 0.15, 'high': 0.25, 'critical': 0.40},
            'leverage_ratio': {'moderate': 1.2, 'high': 1.5, 'critical': 2.0},
            'liquidity_risk': {'moderate': 0.10, 'high': 0.20, 'critical': 0.35}
        }
    
    def calculate_advanced_var(self, returns: pd.Series, confidence_levels: List[float] = None,
                             method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methodologies.
        Supports historical, parametric, and Monte Carlo methods.
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        var_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            if method == 'historical':
                # Historical simulation
                var_value = np.percentile(returns, alpha * 100)
            
            elif method == 'parametric':
                # Parametric (normal distribution assumption)
                mu = returns.mean()
                sigma = returns.std()
                var_value = stats.norm.ppf(alpha, mu, sigma)
            
            elif method == 'monte_carlo':
                # Monte Carlo simulation
                mu = returns.mean()
                sigma = returns.std()
                simulated_returns = np.random.normal(mu, sigma, 10000)
                var_value = np.percentile(simulated_returns, alpha * 100)
            
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            var_results[f'var_{int(confidence*100)}'] = abs(var_value)
        
        return var_results
    
    def calculate_conditional_var(self, returns: pd.Series, 
                                confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        Average loss beyond VaR threshold.
        """
        alpha = 1 - confidence_level
        var_threshold = np.percentile(returns, alpha * 100)
        
        # Calculate average of returns below VaR threshold
        tail_losses = returns[returns <= var_threshold]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        return abs(cvar)
    
    def calculate_maximum_drawdown(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        Returns peak, trough, duration, and recovery information.
        """
        # Calculate cumulative returns
        cum_returns = (1 + price_series.pct_change()).cumprod()
        
        # Calculate running maximum (peak)
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown series
        drawdown = (cum_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_date = running_max.loc[:max_dd_date].idxmax()
        peak_value = running_max.loc[peak_date]
        
        # Find recovery date (if any)
        recovery_date = None
        if max_dd_date < cum_returns.index[-1]:
            post_trough = cum_returns.loc[max_dd_date:]
            recovery_candidates = post_trough[post_trough >= peak_value]
            if len(recovery_candidates) > 0:
                recovery_date = recovery_candidates.index[0]
        
        # Calculate duration
        drawdown_duration = (max_dd_date - peak_date).days
        recovery_duration = None
        if recovery_date:
            recovery_duration = (recovery_date - max_dd_date).days
        
        return {
            'max_drawdown': abs(max_drawdown),
            'peak_date': peak_date,
            'trough_date': max_dd_date,
            'recovery_date': recovery_date,
            'drawdown_duration_days': drawdown_duration,
            'recovery_duration_days': recovery_duration,
            'peak_value': peak_value
        }
    
    def run_stress_test(self, portfolio_weights: Dict[str, float],
                       asset_returns: pd.DataFrame,
                       scenario: StressScenario) -> StressTestResult:
        """
        Run comprehensive stress test on portfolio.
        Applies shock scenarios and measures impact.
        """
        scenario_params = self.stress_scenarios[scenario]
        
        # Apply shocks to asset returns
        stressed_returns = asset_returns.copy()
        
        # Apply equity shock (assume stocks have 'equity' in name or are major indices)
        equity_assets = [col for col in stressed_returns.columns 
                        if any(term in col.upper() for term in ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'])]
        
        bond_assets = [col for col in stressed_returns.columns 
                      if any(term in col.upper() for term in ['BND', 'TLT', 'IEF'])]
        
        # Apply shocks
        for asset in equity_assets:
            if asset in portfolio_weights:
                stressed_returns[asset] += scenario_params['equity_shock'] / 252  # Daily shock
        
        for asset in bond_assets:
            if asset in portfolio_weights:
                stressed_returns[asset] += scenario_params['bond_shock'] / 252  # Daily shock
        
        # Calculate portfolio returns under stress
        portfolio_returns_normal = (asset_returns * pd.Series(portfolio_weights)).sum(axis=1)
        portfolio_returns_stressed = (stressed_returns * pd.Series(portfolio_weights)).sum(axis=1)
        
        # Calculate impact metrics
        portfolio_value_change = portfolio_returns_stressed.sum() - portfolio_returns_normal.sum()
        
        # Individual asset impacts
        individual_impacts = {}
        for asset in portfolio_weights:
            if asset in asset_returns.columns:
                normal_contrib = asset_returns[asset] * portfolio_weights[asset]
                stressed_contrib = stressed_returns[asset] * portfolio_weights[asset]
                individual_impacts[asset] = (stressed_contrib.sum() - normal_contrib.sum())
        
        # Calculate risk metrics under stress
        stressed_var = self.calculate_advanced_var(portfolio_returns_stressed)
        stressed_cvar = self.calculate_conditional_var(portfolio_returns_stressed)
        
        risk_metrics_under_stress = {
            **stressed_var,
            'cvar_95': stressed_cvar,
            'volatility': portfolio_returns_stressed.std() * np.sqrt(252)
        }
        
        # Generate recommendations
        recommendations = self._generate_stress_recommendations(
            portfolio_value_change, individual_impacts, scenario
        )
        
        return StressTestResult(
            scenario=scenario,
            portfolio_value_change=portfolio_value_change,
            individual_asset_impacts=individual_impacts,
            risk_metrics_under_stress=risk_metrics_under_stress,
            probability_of_scenario=self._estimate_scenario_probability(scenario),
            recovery_time_estimate=scenario_params['duration_days'],
            recommended_actions=recommendations
        )
    
    def monitor_portfolio_risk(self, portfolio_id: str, 
                              current_positions: Dict[str, float],
                              market_data: pd.DataFrame) -> List[RiskAlert]:
        """
        Real-time portfolio risk monitoring.
        Generates alerts when risk thresholds are breached.
        """
        alerts = []
        timestamp = datetime.utcnow()
        
        # Calculate current portfolio returns
        portfolio_returns = (market_data * pd.Series(current_positions)).sum(axis=1)
        
        # Check VaR thresholds
        current_var = self.calculate_advanced_var(portfolio_returns)
        for var_type, var_value in current_var.items():
            for threshold_level, threshold_value in self.risk_thresholds['var_95'].items():
                if var_value > threshold_value:
                    alerts.append(RiskAlert(
                        timestamp=timestamp,
                        risk_type=f"Value_at_Risk_{var_type}",
                        severity=RiskLevel(threshold_level),
                        message=f"VaR {var_type} ({var_value:.3f}) exceeds {threshold_level} threshold ({threshold_value:.3f})",
                        portfolio_id=portfolio_id,
                        metric_value=var_value,
                        threshold=threshold_value,
                        action_required=threshold_level in ['high', 'critical']
                    ))
        
        # Check concentration risk
        max_position = max(current_positions.values())
        for threshold_level, threshold_value in self.risk_thresholds['concentration_risk'].items():
            if max_position > threshold_value:
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    risk_type="Concentration_Risk",
                    severity=RiskLevel(threshold_level),
                    message=f"Maximum position size ({max_position:.1%}) exceeds {threshold_level} threshold ({threshold_value:.1%})",
                    portfolio_id=portfolio_id,
                    metric_value=max_position,
                    threshold=threshold_value,
                    action_required=threshold_level in ['high', 'critical']
                ))
        
        # Check liquidity risk (simplified - based on number of positions)
        illiquid_threshold = 0.05  # Positions smaller than 5% considered illiquid
        illiquid_positions = sum(1 for weight in current_positions.values() if weight < illiquid_threshold)
        liquidity_risk_ratio = illiquid_positions / len(current_positions)
        
        for threshold_level, threshold_value in self.risk_thresholds['liquidity_risk'].items():
            if liquidity_risk_ratio > threshold_value:
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    risk_type="Liquidity_Risk",
                    severity=RiskLevel(threshold_level),
                    message=f"Liquidity risk ratio ({liquidity_risk_ratio:.1%}) exceeds {threshold_level} threshold",
                    portfolio_id=portfolio_id,
                    metric_value=liquidity_risk_ratio,
                    threshold=threshold_value,
                    action_required=threshold_level in ['moderate', 'high', 'critical']
                ))
        
        # Store alerts
        self.risk_alerts.extend(alerts)
        
        return alerts
    
    def _generate_stress_recommendations(self, portfolio_impact: float,
                                       asset_impacts: Dict[str, float],
                                       scenario: StressScenario) -> List[str]:
        """Generate actionable recommendations based on stress test results."""
        recommendations = []
        
        if abs(portfolio_impact) > 0.15:  # More than 15% impact
            recommendations.append("Consider reducing overall portfolio risk through diversification")
        
        # Find worst performing assets
        worst_assets = sorted(asset_impacts.items(), key=lambda x: x[1])[:3]
        for asset, impact in worst_assets:
            if impact < -0.05:  # More than 5% negative impact
                recommendations.append(f"Consider reducing exposure to {asset} (stress impact: {impact:.1%})")
        
        # Scenario-specific recommendations
        if scenario == StressScenario.MARKET_CRASH_2008:
            recommendations.append("Increase allocation to government bonds and defensive assets")
            recommendations.append("Consider implementing portfolio insurance strategies")
        
        elif scenario == StressScenario.COVID_PANDEMIC:
            recommendations.append("Increase cash allocation for liquidity during market disruptions")
            recommendations.append("Consider sectors that benefit from economic disruptions (technology, healthcare)")
        
        return recommendations
    
    def _estimate_scenario_probability(self, scenario: StressScenario) -> float:
        """Estimate probability of stress scenario occurrence."""
        # Simplified probability estimates based on historical frequency
        probabilities = {
            StressScenario.MARKET_CRASH_2008: 0.05,  # 5% annual probability
            StressScenario.COVID_PANDEMIC: 0.02,    # 2% annual probability
            StressScenario.DOT_COM_BUBBLE: 0.03,    # 3% annual probability
            StressScenario.CUSTOM: 0.10             # Default for custom scenarios
        }
        return probabilities.get(scenario, 0.05)
    
    def generate_risk_report(self, portfolio_id: str) -> Dict[str, Any]:
        """Generate comprehensive risk assessment report."""
        recent_alerts = [alert for alert in self.risk_alerts 
                        if alert.portfolio_id == portfolio_id and
                        alert.timestamp > datetime.utcnow() - timedelta(days=7)]
        
        # Categorize alerts by severity
        alert_summary = {}
        for level in RiskLevel:
            alert_summary[level.value] = len([a for a in recent_alerts if a.severity == level])
        
        # Risk score calculation (0-100)
        critical_weight = 25
        high_weight = 10
        moderate_weight = 5
        low_weight = 1
        
        risk_score = (
            alert_summary.get('critical', 0) * critical_weight +
            alert_summary.get('high', 0) * high_weight +
            alert_summary.get('moderate', 0) * moderate_weight +
            alert_summary.get('low', 0) * low_weight
        )
        risk_score = min(100, risk_score)  # Cap at 100
        
        # Overall risk level
        if risk_score >= 75:
            overall_risk = RiskLevel.CRITICAL
        elif risk_score >= 50:
            overall_risk = RiskLevel.HIGH
        elif risk_score >= 25:
            overall_risk = RiskLevel.MODERATE
        else:
            overall_risk = RiskLevel.LOW
        
        return {
            'portfolio_id': portfolio_id,
            'assessment_timestamp': datetime.utcnow(),
            'overall_risk_level': overall_risk.value,
            'risk_score': risk_score,
            'alert_summary': alert_summary,
            'recent_alerts_count': len(recent_alerts),
            'active_monitoring': self.monitoring_active,
            'recommendations': self._generate_general_risk_recommendations(overall_risk, alert_summary)
        }
    
    def _generate_general_risk_recommendations(self, risk_level: RiskLevel, 
                                             alert_summary: Dict) -> List[str]:
        """Generate general risk management recommendations."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Immediate portfolio review required",
                "Consider reducing position sizes across all assets",
                "Implement stop-loss orders for high-risk positions",
                "Increase cash allocation for liquidity"
            ])
        
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Schedule portfolio rebalancing within 48 hours",
                "Review and tighten risk parameters",
                "Consider hedging strategies for downside protection"
            ])
        
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "Monitor portfolio closely for trend changes",
                "Review correlation between major positions",
                "Consider gradual rebalancing if trends continue"
            ])
        
        return recommendations

# Example usage demonstration
def demonstrate_advanced_risk_management():
    """Demonstrate advanced risk management capabilities."""
    
    # Initialize risk engine
    risk_engine = EnterpriseRiskEngine()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Simulate asset returns
    assets = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'BND']
    returns_data = {}
    
    for asset in assets:
        # Generate correlated returns with different volatilities
        base_return = 0.0003 + np.random.normal(0, 0.015, len(dates))  # Daily returns
        if asset == 'BND':  # Lower volatility for bonds
            base_return *= 0.3
        returns_data[asset] = base_return
    
    asset_returns = pd.DataFrame(returns_data, index=dates)
    
    # Sample portfolio
    portfolio_weights = {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.15, 'SPY': 0.30, 'BND': 0.10}
    
    print("Advanced Risk Management Demonstration:")
    
    # Calculate advanced VaR
    portfolio_returns = (asset_returns * pd.Series(portfolio_weights)).sum(axis=1)
    var_results = risk_engine.calculate_advanced_var(portfolio_returns)
    print(f"\nValue at Risk Results:")
    for var_type, value in var_results.items():
        print(f"{var_type}: {value:.4f}")
    
    # Calculate CVaR
    cvar = risk_engine.calculate_conditional_var(portfolio_returns)
    print(f"Conditional VaR (95%): {cvar:.4f}")
    
    # Run stress test
    stress_result = risk_engine.run_stress_test(
        portfolio_weights, asset_returns, StressScenario.COVID_PANDEMIC
    )
    print(f"\nCOVID-19 Stress Test Results:")
    print(f"Portfolio Impact: {stress_result.portfolio_value_change:.2%}")
    print(f"Scenario Probability: {stress_result.probability_of_scenario:.1%}")
    print(f"Recovery Time Estimate: {stress_result.recovery_time_estimate} days")
    
    # Monitor portfolio risk
    alerts = risk_engine.monitor_portfolio_risk('DEMO_001', portfolio_weights, asset_returns)
    print(f"\nRisk Monitoring: {len(alerts)} alerts generated")
    
    # Generate risk report
    risk_report = risk_engine.generate_risk_report('DEMO_001')
    print(f"\nRisk Assessment Summary:")
    print(f"Overall Risk Level: {risk_report['overall_risk_level']}")
    print(f"Risk Score: {risk_report['risk_score']}/100")
