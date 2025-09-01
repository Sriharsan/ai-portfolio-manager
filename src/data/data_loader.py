# src/data/data_loader.py

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from data.market_data import market_data_provider
from models.llm_engine import llm_engine
from analytics.performance import performance_analyzer
from models.risk_manager import risk_manager
from models.portfolio_optimizer import InstitutionalPortfolioBuilder

class DataLoader:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_portfolio_analysis(
        self, 
        portfolio_weights: Dict[str, float],
        period: str = '1y'
    ) -> Dict:
        
        try:
            # Get market data
            stock_data, portfolio_returns = market_data_provider.get_portfolio_data(
                portfolio_weights, period
            )
            
            # Performance metrics
            returns_series = portfolio_returns['Daily_Return']
            performance_metrics = performance_analyzer.calculate_metrics(returns_series)
            
            # Risk metrics
            risk_metrics = risk_manager.portfolio_risk_metrics(returns_series)
            
            # Optimization recommendations
            returns_df = pd.DataFrame({
                symbol: stock_data[symbol]['Daily_Return'] 
                for symbol in portfolio_weights.keys() 
                if symbol in stock_data
            }).dropna()
            
            if len(returns_df) > 0:
                portfolio_builder = InstitutionalPortfolioBuilder(market_data_provider)
                optimization_result = portfolio_builder.build_portfolio_from_template('balanced')
            else:
                optimization_result = None            
            
            # AI insights
            ai_summary = llm_engine.generate_portfolio_summary(stock_data, portfolio_weights)
            
            return {
                'portfolio_data': portfolio_returns,
                'stock_data': stock_data,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'optimization': optimization_result,
                'ai_insights': ai_summary,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {e}")
            return {'error': str(e)}
    
    def get_stock_analysis(self, symbol: str, period: str = '6mo') -> Dict:
        
        try:
            # Get stock data
            stock_data = market_data_provider.get_stock_data(symbol, period)
            
            # Performance analysis
            returns = stock_data['Daily_Return']
            performance = performance_analyzer.calculate_metrics(returns)
            
            # AI insight
            ai_insight = llm_engine.generate_market_insight(stock_data, symbol)
            
            # Risk assessment
            risk_assessment = risk_manager.portfolio_risk_metrics(returns)
            
            return {
                'stock_data': stock_data,
                'performance': performance,
                'ai_insight': ai_insight,
                'risk_metrics': risk_assessment,
                'current_price': stock_data['Close'].iloc[-1],
                'price_change': stock_data['Daily_Return'].iloc[-1] * 100
            }
            
        except Exception as e:
            self.logger.error(f"Stock analysis failed for {symbol}: {e}")
            return {'error': str(e)}

# Global instance
data_loader = DataLoader()