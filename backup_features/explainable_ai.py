import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PortfolioExplainer:
    """
    Comprehensive explainability engine for portfolio management decisions.
    Integrates SHAP and LIME for different types of explanations.
    """
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = []
        self.explanation_cache = {}
    
    def initialize_explainers(self, model, training_data: np.ndarray, 
                            feature_names: List[str]):
        """Initialize SHAP and LIME explainers with trained model."""
        self.feature_names = feature_names
        
        # Initialize SHAP explainer
        if hasattr(model, 'predict_proba'):
            # For classification models
            self.shap_explainer = shap.Explainer(model, training_data)
        else:
            # For regression models
            self.shap_explainer = shap.Explainer(model.predict, training_data)
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode='regression',
            verbose=True,
            random_state=42
        )
    
    def explain_portfolio_allocation(self, portfolio_weights: Dict[str, float],
                                   features: np.ndarray, model) -> Dict:
        """
        Explain why specific portfolio allocations were chosen.
        Returns detailed attribution for each asset weight.
        """
        explanations = {}
        
        # SHAP explanations
        if self.shap_explainer:
            shap_values = self.shap_explainer(features.reshape(1, -1))
            
            # Map SHAP values to portfolio decisions
            for i, asset in enumerate(portfolio_weights.keys()):
                if i < len(shap_values.values[0]):
                    explanations[f"{asset}_shap_attribution"] = {
                        'value': float(shap_values.values[0][i]),
                        'base_value': float(shap_values.base_values[0]),
                        'feature_contributions': {
                            self.feature_names[j]: float(shap_values.values[0][j])
                            for j in range(len(self.feature_names))
                        }
                    }
        
        # LIME explanations for individual assets
        for asset in portfolio_weights.keys():
            if self.lime_explainer:
                # Create asset-specific prediction function
                def asset_prediction_fn(X):
                    predictions = model.predict(X)
                    # Return prediction for this specific asset (simplified)
                    return predictions.reshape(-1, 1)
                
                explanation = self.lime_explainer.explain_instance(
                    features,
                    asset_prediction_fn,
                    num_features=len(self.feature_names)
                )
                
                explanations[f"{asset}_lime_explanation"] = {
                    'feature_importance': explanation.as_list(),
                    'score': explanation.score,
                    'intercept': explanation.intercept[0] if explanation.intercept else 0
                }
        
        return explanations
    
    def explain_risk_metrics(self, risk_metrics: Dict[str, float],
                           portfolio_data: pd.DataFrame) -> Dict:
        """
        Explain how portfolio composition contributes to risk metrics.
        Provides breakdown of VaR, Sharpe ratio, etc. by asset/sector.
        """
        explanations = {}
        
        # Asset contribution to portfolio risk
        if 'volatility' in risk_metrics and len(portfolio_data) > 1:
            returns = portfolio_data.pct_change().dropna()
            
            # Calculate marginal contribution to risk
            portfolio_returns = returns.mean(axis=1)  # Equal-weighted for simplicity
            portfolio_vol = portfolio_returns.std()
            
            marginal_contributions = {}
            for asset in returns.columns:
                # Calculate beta with portfolio
                asset_returns = returns[asset]
                covariance = np.cov(asset_returns, portfolio_returns)[0, 1]
                marginal_contrib = covariance / portfolio_vol if portfolio_vol > 0 else 0
                marginal_contributions[asset] = marginal_contrib
            
            explanations['risk_attribution'] = marginal_contributions
        
        # Sharpe ratio breakdown
        if 'sharpe_ratio' in risk_metrics:
            # Simplified attribution (in practice would use more sophisticated methods)
            explanations['sharpe_components'] = {
                'excess_return': risk_metrics.get('expected_return', 0) - 0.02,  # Assume 2% risk-free rate
                'volatility': risk_metrics.get('volatility', 0),
                'risk_adjusted_performance': risk_metrics['sharpe_ratio']
            }
        
        return explanations
    
    def generate_explanation_report(self, portfolio_weights: Dict[str, float],
                                  risk_metrics: Dict[str, float],
                                  explanations: Dict) -> str:
        """
        Generate human-readable explanation report for portfolio decisions.
        Suitable for client presentations or regulatory documentation.
        """
        report = []
        report.append("=== PORTFOLIO DECISION EXPLANATION REPORT ===\n")
        
        # Portfolio composition explanation
        report.append("ASSET ALLOCATION RATIONALE:")
        for asset, weight in portfolio_weights.items():
            shap_key = f"{asset}_shap_attribution"
            if shap_key in explanations:
                attribution = explanations[shap_key]
                top_features = sorted(
                    attribution['feature_contributions'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:3]
                
                report.append(f"\n{asset} ({weight:.1%} allocation):")
                report.append(f"  - Model confidence: {attribution['value']:.3f}")
                report.append("  - Key driving factors:")
                for feature, contrib in top_features:
                    direction = "positive" if contrib > 0 else "negative"
                    report.append(f"    * {feature}: {direction} impact ({contrib:.3f})")
        
        # Risk explanation
        if 'risk_attribution' in explanations:
            report.append("\n\nRISK ATTRIBUTION ANALYSIS:")
            risk_contrib = explanations['risk_attribution']
            for asset, contrib in sorted(risk_contrib.items(), 
                                       key=lambda x: abs(x[1]), reverse=True):
                risk_level = "high" if abs(contrib) > 0.1 else "moderate" if abs(contrib) > 0.05 else "low"
                report.append(f"  - {asset}: {risk_level} risk contribution ({contrib:.3f})")
        
        # Performance metrics explanation
        if 'sharpe_components' in explanations:
            components = explanations['sharpe_components']
            report.append(f"\n\nRISK-ADJUSTED PERFORMANCE BREAKDOWN:")
            report.append(f"  - Expected excess return: {components['excess_return']:.2%}")
            report.append(f"  - Portfolio volatility: {components['volatility']:.2%}")
            report.append(f"  - Sharpe ratio: {components['risk_adjusted_performance']:.2f}")
        
        return "\n".join(report)
    
    def create_explanation_visualizations(self, explanations: Dict) -> Dict[str, go.Figure]:
        """Create interactive visualizations for explanations."""
        figures = {}
        
        # SHAP feature importance plot
        if any('shap_attribution' in key for key in explanations.keys()):
            # Collect SHAP values for all assets
            shap_data = {}
            for key, value in explanations.items():
                if 'shap_attribution' in key:
                    asset = key.replace('_shap_attribution', '')
                    shap_data[asset] = value['feature_contributions']
            
            if shap_data:
                # Create feature importance heatmap
                assets = list(shap_data.keys())
                features = list(next(iter(shap_data.values())).keys())
                
                z_values = []
                for asset in assets:
                    z_values.append([shap_data[asset][feature] for feature in features])
                
                fig = go.Figure(data=go.Heatmap(
                    z=z_values,
                    x=features,
                    y=assets,
                    colorscale='RdBu',
                    zmid=0,
                    text=[[f"{val:.3f}" for val in row] for row in z_values],
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="SHAP Feature Importance by Asset",
                    xaxis_title="Features",
                    yaxis_title="Assets",
                    height=400
                )
                figures['shap_heatmap'] = fig
        
        # Risk attribution pie chart
        if 'risk_attribution' in explanations:
            risk_data = explanations['risk_attribution']
            
            # Convert to absolute values for pie chart
            abs_contributions = {k: abs(v) for k, v in risk_data.items()}
            total = sum(abs_contributions.values())
            
            if total > 0:
                labels = list(abs_contributions.keys())
                values = [v/total * 100 for v in abs_contributions.values()]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='inside'
                )])
                
                fig.update_layout(
                    title="Risk Contribution by Asset",
                    height=400
                )
                figures['risk_pie'] = fig
        
        return figures

# Example usage for interview demonstration
def demonstrate_explainable_ai():
    """Demonstrate explainable AI capabilities."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    feature_names = [
        'price_momentum', 'volume_trend', 'volatility', 'market_cap',
        'pe_ratio', 'dividend_yield', 'beta', 'rsi', 'macd', 'sentiment'
    ]
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 0.3 + X[:, 2] * -0.2 + X[:, 5] * 0.1 + 
         np.random.randn(n_samples) * 0.1)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize explainer
    explainer = PortfolioExplainer()
    explainer.initialize_explainers(model, X_train, feature_names)
    
    # Sample portfolio and explanation
    portfolio_weights = {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.15}
    sample_features = X_test[0]  # Use first test sample
    
    explanations = explainer.explain_portfolio_allocation(
        portfolio_weights, sample_features, model
    )
    
    # Generate report
    risk_metrics = {'sharpe_ratio': 1.24, 'volatility': 0.15, 'expected_return': 0.08}
    report = explainer.generate_explanation_report(
        portfolio_weights, risk_metrics, explanations
    )
    
    print("Explainable AI Demonstration:")
    print(report)