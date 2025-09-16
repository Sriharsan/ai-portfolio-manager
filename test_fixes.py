# test_fixes.py

import logging
from src.models.llm_engine import OptimizedLLMEngine
from src.models.risk_manager import create_risk_manager
from src.data.market_data import market_data_provider  # ✅ fixed import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize Market Data Provider
    market_data = market_data_provider
    logger.info("✅ Market Data Provider loaded successfully")

    # Initialize Risk Manager
    risk_manager = create_risk_manager()
    logger.info("✅ Risk Manager loaded successfully")

    # Initialize LLM Engine
    llm_engine = OptimizedLLMEngine()
    logger.info("✅ LLM Engine loaded successfully")

    # Quick smoke test: fetch data for AAPL
    try:
        data = market_data.get_stock_data("AAPL", "1mo")
        logger.info(f"📈 AAPL data sample:\n{data.head()}")
    except Exception as e:
        logger.error(f"Market data fetch failed: {e}")

    # Quick risk test with dummy returns
    import pandas as pd
    import numpy as np

    dummy_returns = pd.Series(np.random.normal(0, 0.01, 252))
    # Test the correct methods that exist in DynamicRiskManager
    try:
        # Test basic VaR calculation
        var_results = risk_manager.calculate_var(dummy_returns)
        logger.info(f"📊 VaR results: {var_results}")
    
        # Test comprehensive analysis
        comprehensive_results = risk_manager.comprehensive_portfolio_analysis(dummy_returns)
        logger.info(f"📈 Comprehensive analysis completed: {list(comprehensive_results.keys())}")
    
        # Test max drawdown
        max_dd = risk_manager.max_drawdown(dummy_returns)
        logger.info(f"📉 Max Drawdown: {max_dd:.4f}")
    
    except Exception as e:
        logger.error(f"Risk manager test failed: {e}")