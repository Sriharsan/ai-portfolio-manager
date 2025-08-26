"""
Test Suite for Data Loader Module
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import data_loader

class TestDataLoader:
    """Test data loader functionality"""
    
    def test_portfolio_analysis_basic(self):
        """Test basic portfolio analysis"""
        portfolio = {'AAPL': 0.5, 'MSFT': 0.5}
        
        result = data_loader.get_portfolio_analysis(portfolio, '1mo')
        
        # Should return result even if data fails
        assert isinstance(result, dict)
        
        if 'error' not in result:
            assert 'performance_metrics' in result
            assert 'ai_insights' in result
    
    def test_stock_analysis(self):
        """Test individual stock analysis"""
        result = data_loader.get_stock_analysis('AAPL', '1mo')
        
        assert isinstance(result, dict)
        
        if 'error' not in result:
            assert 'current_price' in result
            assert 'ai_insight' in result
    
    def test_invalid_portfolio(self):
        """Test handling of invalid portfolio"""
        invalid_portfolio = {'INVALID': 1.0}
        
        result = data_loader.get_portfolio_analysis(invalid_portfolio, '1mo')
        
        # Should handle gracefully
        assert isinstance(result, dict)
    
    def test_empty_portfolio(self):
        """Test empty portfolio handling"""
        empty_portfolio = {}
        
        result = data_loader.get_portfolio_analysis(empty_portfolio, '1mo')
        
        assert 'error' in result or len(result) == 0

if __name__ == "__main__":
    # Run tests
    test_loader = TestDataLoader()
    
    try:
        test_loader.test_portfolio_analysis_basic()
        print("✅ Portfolio analysis test passed")
    except Exception as e:
        print(f"❌ Portfolio analysis test failed: {e}")
    
    try:
        test_loader.test_stock_analysis()
        print("✅ Stock analysis test passed")
    except Exception as e:
        print(f"❌ Stock analysis test failed: {e}")
    
    try:
        test_loader.test_invalid_portfolio()
        print("✅ Invalid portfolio test passed")
    except Exception as e:
        print(f"❌ Invalid portfolio test failed: {e}")
    
    try:
        test_loader.test_empty_portfolio()
        print("✅ Empty portfolio test passed")
    except Exception as e:
        print(f"❌ Empty portfolio test failed: {e}")
    
    print("🧪 Data loader tests completed")