"""
Market Data Provider - Fixed Import Issues
Optimized for production deployment
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import time
from functools import wraps
import logging
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Optimized market data provider"""
    
    def __init__(self):
        # Import config safely
        try:
            from config import config
            self.alpha_vantage_key = getattr(config, 'ALPHA_VANTAGE_API_KEY', '')
            self.fred_key = getattr(config, 'FRED_API_KEY', '')
            self.cache_dir = getattr(config, 'CACHE_DIR', Path('data/cache'))
        except ImportError:
            logger.warning("Config import failed, using defaults")
            self.alpha_vantage_key = ''
            self.fred_key = ''
            self.cache_dir = Path('data/cache')
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 1.0  # 1 second between calls
        
        logger.info("Market Data Provider initialized")
    
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        
        self.last_call_time = time.time()
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get stock data with fallback handling"""
        
        self._rate_limit()
        
        try:
            logger.info(f"Fetching {symbol} data for {period}")
            
            # Use yfinance as primary source
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return self._create_mock_data(symbol, period)
            
            # Clean and enhance data
            data = self._clean_stock_data(data)
            data = self._add_technical_indicators(data)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return self._create_mock_data(symbol, period)
    
    def _clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean stock data"""
        # Remove any rows with NaN in essential columns
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Ensure OHLC relationships are correct
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        
        # Daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Moving averages
        if len(data) >= 20:
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
        if len(data) >= 50:
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI (simplified)
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if len(data) >= 20:
            sma20 = data['Close'].rolling(window=20).mean()
            std20 = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = sma20 + (std20 * 2)
            data['BB_Lower'] = sma20 - (std20 * 2)
        
        return data
    
    def _create_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Create mock data when real data unavailable"""
        
        logger.info(f"Creating mock data for {symbol}")
        
        # Determine date range
        end_date = datetime.now()
        if period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=365)
        
        # Generate dates (business days only)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        if n_days == 0:
            return pd.DataFrame()
        
        # Set random seed for reproducible mock data
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate realistic stock data
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.015, n_days)  # Daily returns
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.005, 0.02)
            high = close * (1 + volatility/2)
            low = close * (1 - volatility/2)
            open_price = close + np.random.normal(0, close * 0.003)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks"""
        
        logger.info(f"Fetching data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                # Continue with other symbols
                
        return results
    
    def get_portfolio_data(self, portfolio: Dict[str, float], period: str = "1y") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Get portfolio data and calculate returns"""
        
        # Validate portfolio weights
        total_weight = sum(portfolio.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Portfolio weights sum to {total_weight:.3f}, not 1.0")
        
        # Get individual stock data
        symbols = list(portfolio.keys())
        stock_data = self.get_multiple_stocks(symbols, period)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(stock_data, portfolio)
        
        return stock_data, portfolio_returns
    
    def _calculate_portfolio_returns(self, stock_data: Dict[str, pd.DataFrame], portfolio: Dict[str, float]) -> pd.DataFrame:
        """Calculate weighted portfolio returns"""
        
        if not stock_data:
            return pd.DataFrame()
        
        # Get common date range
        all_dates = set()
        for data in stock_data.values():
            if not data.empty:
                all_dates.update(data.index)
        
        if not all_dates:
            return pd.DataFrame()
        
        all_dates = sorted(all_dates)
        
        # Calculate daily portfolio returns
        portfolio_data = pd.DataFrame(index=all_dates)
        daily_returns = []
        
        for date in all_dates:
            daily_return = 0.0
            total_weight = 0.0
            
            for symbol, weight in portfolio.items():
                if symbol in stock_data and date in stock_data[symbol].index:
                    stock_return = stock_data[symbol].loc[date, 'Daily_Return']
                    if not pd.isna(stock_return):
                        daily_return += weight * stock_return
                        total_weight += weight
            
            # Normalize by actual available weight
            if total_weight > 0:
                daily_return = daily_return / total_weight
            
            daily_returns.append(daily_return)
        
        portfolio_data['Daily_Return'] = daily_returns
        portfolio_data['Cumulative_Return'] = (1 + portfolio_data['Daily_Return']).cumprod()
        
        # Portfolio value (assuming $100,000 initial investment)
        initial_value = 100000
        portfolio_data['Portfolio_Value'] = initial_value * portfolio_data['Cumulative_Return']
        
        # Add rolling statistics
        if len(portfolio_data) >= 30:
            portfolio_data['Volatility_30d'] = portfolio_data['Daily_Return'].rolling(30).std() * np.sqrt(252)
            portfolio_data['Sharpe_30d'] = (portfolio_data['Daily_Return'].rolling(30).mean() * 252) / (portfolio_data['Daily_Return'].rolling(30).std() * np.sqrt(252))
        
        # Drawdown
        rolling_max = portfolio_data['Portfolio_Value'].cummax()
        portfolio_data['Drawdown'] = (portfolio_data['Portfolio_Value'] - rolling_max) / rolling_max
        
        return portfolio_data

# Global instance
market_data = MarketDataProvider()