# tests/test_analytics.py

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from analytics.performance import performance_analyzer
from analytics.attribution import attribution_analyzer
from analytics.reporting import report_generator

class TestAnalytics:
    
    def setup_method(self):
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),
            index=pd.date_range('2023-01-01', periods=252)
        )
        self.benchmark = pd.Series(
            np.random.normal(0.0008, 0.015, 252),
            index=pd.date_range('2023-01-01', periods=252)
        )
    
    def test_performance_metrics(self):
        metrics = performance_analyzer.calculate_metrics(self.returns)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'volatility' in metrics
        assert 'max_drawdown' in metrics
        
        # Check reasonable values
        assert -1 <= metrics['total_return'] <= 2
        assert -2 <= metrics['sharpe_ratio'] <= 5
        assert 0 <= metrics['volatility'] <= 1
    
    def test_performance_with_benchmark(self):
        metrics = performance_analyzer.calculate_metrics(self.returns, self.benchmark)
        
        assert 'benchmark_return' in metrics
        assert 'excess_return' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics
    
    def test_rolling_metrics(self):
        rolling_data = performance_analyzer.rolling_metrics(self.returns, window=60)
        
        assert len(rolling_data) > 0
        assert 'rolling_return' in rolling_data.columns
        assert 'rolling_sharpe' in rolling_data.columns
    
    def test_attribution_analysis(self):
        portfolio_returns = {
            'AAPL': self.returns,
            'MSFT': self.returns * 1.1
        }
        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        attribution = attribution_analyzer.sector_attribution(
            portfolio_returns, weights, self.benchmark
        )
        
        assert isinstance(attribution, dict)
        assert 'AAPL' in attribution
        assert 'MSFT' in attribution
    
    def test_report_generation(self):
        analysis_data = {
            'performance_metrics': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'volatility': 0.18
            },
            'risk_metrics': {
                'var_1d_95': -2000,
                'cvar_1d_95': -2500
            }
        }
        
        report = report_generator.generate_summary_report(analysis_data)
        
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert '15.0%' in report  # Should contain formatted metrics
    
    def test_insufficient_data_handling(self):
        short_returns = pd.Series([0.01])
        
        metrics = performance_analyzer.calculate_metrics(short_returns)
        
        assert 'error' in metrics

if __name__ == "__main__":

    test_analytics = TestAnalytics()
    test_analytics.setup_method()
    
    tests = [
        ('performance_metrics', test_analytics.test_performance_metrics),
        ('performance_with_benchmark', test_analytics.test_performance_with_benchmark),
        ('rolling_metrics', test_analytics.test_rolling_metrics),
        ('attribution_analysis', test_analytics.test_attribution_analysis),
        ('report_generation', test_analytics.test_report_generation),
        ('insufficient_data_handling', test_analytics.test_insufficient_data_handling)
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} test passed")
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
    
    print("🧪 Analytics tests completed")