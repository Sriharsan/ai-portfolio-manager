# tests/test_models.py

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.portfolio_optimizer import optimizer
from models.risk_manager import risk_manager
from models.sentiment_analyzer import sentiment_analyzer
from analytics.performance import performance_analyzer
import pandas as pd
import numpy as np

def test_portfolio_optimizer():
    
    # Generate sample return data
    dates = pd.date_range('2023-01-01', periods=100)
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 100),
        'MSFT': np.random.normal(0.0008, 0.018, 100),
        'GOOGL': np.random.normal(0.0012, 0.025, 100)
    }, index=dates)
    
    # Test optimization
    result = optimizer.optimize_portfolio(returns, method='max_sharpe')
    
    assert 'weights' in result
    assert abs(sum(result['weights'].values()) - 1.0) < 0.01
    assert result['optimization_success']

def test_risk_manager():
    
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    metrics = risk_manager.portfolio_risk_metrics(returns)
    
    assert 'var_1d_95' in metrics
    assert 'cvar_1d_95' in metrics
    assert metrics['var_1d_95'] < 0  # VaR should be negative

def test_sentiment_analyzer():
    
    positive_text = "Strong earnings beat expectations with bullish outlook"
    negative_text = "Declining revenue amid bearish market conditions"
    
    pos_result = sentiment_analyzer.analyze_sentiment(positive_text)
    neg_result = sentiment_analyzer.analyze_sentiment(negative_text)
    
    assert pos_result['label'] == 'positive'
    assert neg_result['label'] == 'negative'

def test_performance_analyzer():
    
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    metrics = performance_analyzer.calculate_metrics(returns)
    
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'volatility' in metrics

if __name__ == "__main__":
    test_portfolio_optimizer()
    test_risk_manager()
    test_sentiment_analyzer()
    test_performance_analyzer()
    print("All tests passed!")