# tests/test_visualization.py

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from visualization.charts import chart_generator
from visualization.dashboards import dashboard

class TestVisualization:
    
    def setup_method(self):
        dates = pd.date_range('2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Daily_Return': np.random.normal(0.001, 0.02, 100),
            'Cumulative_Return': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            'Portfolio_Value': 100000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        }, index=dates)
        
        self.portfolio_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
    
    def test_performance_chart(self):
        fig = chart_generator.create_performance_chart(self.test_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have at least one trace
        assert fig.layout.title.text is not None
    
    def test_allocation_pie_chart(self):
        fig = chart_generator.create_allocation_pie(self.portfolio_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_risk_metrics_bar_chart(self):
        metrics = {
            'var_1d_95': -2000,
            'cvar_1d_95': -2500,
            'maximum_drawdown': -0.15,
            'volatility_annual': 0.18
        }
        
        fig = chart_generator.create_risk_metrics_bar(metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_correlation_heatmap(self):
        returns_data = {
            'AAPL': pd.Series(np.random.normal(0.001, 0.02, 100)),
            'MSFT': pd.Series(np.random.normal(0.0008, 0.018, 100)),
            'GOOGL': pd.Series(np.random.normal(0.0012, 0.025, 100))
        }
        
        fig = chart_generator.create_correlation_heatmap(returns_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_portfolio_comparison_chart(self):
        portfolios = {
            'Portfolio A': self.test_data,
            'Portfolio B': self.test_data * 1.1  # Slightly different performance
        }
        
        fig = dashboard.portfolio_comparison_chart(portfolios)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Should have two traces
    
    def test_drawdown_chart(self):
        returns = self.test_data['Daily_Return']
        
        fig = dashboard.drawdown_chart(returns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_gauge_chart(self):
        fig = dashboard.create_gauge_chart(75, "Risk Score", 0, 100)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_empty_data_handling(self):
        empty_data = pd.DataFrame()
        
        # Should not raise exceptions
        fig = chart_generator.create_performance_chart(empty_data)
        assert isinstance(fig, go.Figure)

if __name__ == "__main__":

    test_viz = TestVisualization()
    test_viz.setup_method()
    
    tests = [
        ('performance_chart', test_viz.test_performance_chart),
        ('allocation_pie_chart', test_viz.test_allocation_pie_chart),
        ('risk_metrics_bar_chart', test_viz.test_risk_metrics_bar_chart),
        ('correlation_heatmap', test_viz.test_correlation_heatmap),
        ('portfolio_comparison_chart', test_viz.test_portfolio_comparison_chart),
        ('drawdown_chart', test_viz.test_drawdown_chart),
        ('gauge_chart', test_viz.test_gauge_chart),
        ('empty_data_handling', test_viz.test_empty_data_handling)
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} test passed")
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
    
    print("🧪 Visualization tests completed")