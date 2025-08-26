"""
AI Portfolio Management System - Market Data Module
Real-time financial data integration with multiple providers
Enterprise-grade data pipeline with caching and error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Union, Tuple
import time
from functools import wraps
import logging
from pathlib import Path
import pickle

# Import configuration
from config import config

class MarketDataError(Exception):
    """Custom exception for market data errors"""
    pass

def rate_limit(calls_per_minute: int = 60):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 60.0 / calls_per_minute - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def cache_data(cache_duration_hours: int = 1):
    """Data caching decorator"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Check if cached data exists and is still valid
            if cache_key in cache:
                cached_data, cached_time = cache[cache_key]
                if datetime.now() - cached_time < timedelta(hours=cache_duration_hours):
                    logging.info(f"Using cached data for {func.__name__}")
                    return cached_data
            
            # Fetch fresh data
            result = func(*args, **kwargs)
            cache[cache_key] = (result, datetime.now())
            return result
        return wrapper
    return decorator

class MarketDataProvider:
    """
    Enterprise-grade market data provider with multiple data sources
    Handles real-time and historical financial data
    """
    
    def __init__(self):
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        self.fred_key = config.FRED_API_KEY
        self.cache_dir = config.DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)
        
        # Data source priorities (fallback order)
        self.data_sources = ['yfinance', 'alpha_vantage', 'manual']
        
        self.logger.info("🚀 Market Data Provider initialized")
    
    @cache_data(cache_duration_hours=1)
    @rate_limit(calls_per_minute=60)
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get stock data with automatic fallback between data sources
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        
        for source in self.data_sources:
            try:
                if source == 'yfinance':
                    return self._get_yfinance_data(symbol, period, interval)
                elif source == 'alpha_vantage':
                    return self._get_alpha_vantage_data(symbol, period)
                elif source == 'manual':
                    return self._get_manual_data(symbol, period)
            except Exception as e:
                self.logger.warning(f"Failed to fetch data from {source}: {str(e)}")
                continue
        
        raise MarketDataError(f"Failed to fetch data for {symbol} from all sources")
    
    def _get_yfinance_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data using yfinance"""
        self.logger.info(f"📈 Fetching {symbol} data from Yahoo Finance")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise MarketDataError(f"No data found for symbol {symbol}")
        
        # Standardize column names
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(data.columns[5:])
        data.index.name = 'Date'
        
        # Add basic technical indicators
        data = self._add_technical_indicators(data)
        
        self.logger.info(f"✅ Successfully fetched {len(data)} records for {symbol}")
        return data
    
    def _get_alpha_vantage_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data using Alpha Vantage API"""
        if not self.alpha_vantage_key:
            raise MarketDataError("Alpha Vantage API key not configured")
        
        self.logger.info(f"📊 Fetching {symbol} data from Alpha Vantage")
        
        # Map period to Alpha Vantage function
        if period in ['1d', '5d']:
            function = 'TIME_SERIES_INTRADAY'
            interval = '60min'
        else:
            function = 'TIME_SERIES_DAILY'
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full'
        }
        
        if function == 'TIME_SERIES_INTRADAY':
            params['interval'] = interval
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Handle API errors
        if 'Error Message' in data:
            raise MarketDataError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'Note' in data:
            raise MarketDataError("Alpha Vantage API rate limit exceeded")
        
        # Parse the data
        if function == 'TIME_SERIES_INTRADAY':
            time_series = data.get('Time Series (60min)', {})
        else:
            time_series = data.get('Time Series (Daily)', {})
        
        if not time_series:
            raise MarketDataError(f"No time series data found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        
        # Standardize column names
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High', 
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        df = df.sort_index()
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        self.logger.info(f"✅ Successfully fetched {len(df)} records for {symbol}")
        return df
    
    def _get_manual_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate synthetic data for testing (fallback)"""
        self.logger.warning(f"⚠️ Generating synthetic data for {symbol}")
        
        # Generate date range
        end_date = datetime.now()
        if period == '1d':
            start_date = end_date - timedelta(days=1)
        elif period == '5d':
            start_date = end_date - timedelta(days=5)
        elif period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=365)
        
        # Generate synthetic OHLCV data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        np.random.seed(42)  # For reproducible data
        
        # Start with a base price
        base_price = 100.0
        
        # Generate random walk for prices
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = np.random.uniform(0.005, 0.03)
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close + np.random.normal(0, close * 0.005)
            
            # Ensure OHLC relationships are correct
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.randint(1000000, 10000000)
            
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
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data"""
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Price change indicators
        df['Daily_Return'] = df['Close'].pct_change()
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
        
        return df
    
    @cache_data(cache_duration_hours=24)
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks efficiently
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        
        self.logger.info(f"📊 Fetching data for {len(symbols)} symbols")
        
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period)
                results[symbol] = data
                self.logger.info(f"✅ {symbol}: {len(data)} records")
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch {symbol}: {str(e)}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            self.logger.warning(f"⚠️ Failed to fetch data for: {failed_symbols}")
        
        return results
    
    def get_portfolio_data(
        self, 
        portfolio: Dict[str, float], 
        period: str = "1y"
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Get data for an entire portfolio and calculate portfolio returns
        
        Args:
            portfolio: Dictionary mapping symbols to weights (should sum to 1.0)
            period: Data period
            
        Returns:
            Tuple of (individual stock data, portfolio returns)
        """
        
        self.logger.info(f"🎯 Fetching portfolio data for {len(portfolio)} assets")
        
        # Validate portfolio weights
        total_weight = sum(portfolio.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"⚠️ Portfolio weights sum to {total_weight:.3f}, not 1.0")
        
        # Get data for all symbols
        symbols = list(portfolio.keys())
        stock_data = self.get_multiple_stocks(symbols, period)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(stock_data, portfolio)
        
        return stock_data, portfolio_returns
    
    def _calculate_portfolio_returns(
        self, 
        stock_data: Dict[str, pd.DataFrame], 
        portfolio: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate portfolio returns from individual stock data"""
        
        # Get all available dates
        all_dates = set()
        for data in stock_data.values():
            all_dates.update(data.index)
        
        all_dates = sorted(all_dates)
        
        # Create portfolio returns DataFrame
        portfolio_data = pd.DataFrame(index=all_dates)
        
        # Calculate weighted returns
        weighted_returns = []
        
        for date in all_dates:
            daily_return = 0.0
            total_weight = 0.0
            
            for symbol, weight in portfolio.items():
                if symbol in stock_data and date in stock_data[symbol].index:
                    stock_return = stock_data[symbol].loc[date, 'Daily_Return']
                    if not pd.isna(stock_return):
                        daily_return += weight * stock_return
                        total_weight += weight
            
            # Normalize by actual weight (in case some stocks missing data)
            if total_weight > 0:
                daily_return = daily_return / total_weight
            
            weighted_returns.append(daily_return)
        
        portfolio_data['Daily_Return'] = weighted_returns
        portfolio_data['Cumulative_Return'] = (1 + portfolio_data['Daily_Return']).cumprod()
        
        # Calculate portfolio value (assuming $100,000 initial investment)
        initial_value = 100000
        portfolio_data['Portfolio_Value'] = initial_value * portfolio_data['Cumulative_Return']
        
        # Add portfolio statistics
        portfolio_data = self._add_portfolio_statistics(portfolio_data)
        
        return portfolio_data
    
    def _add_portfolio_statistics(self, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical measures to portfolio data"""
        
        returns = portfolio_data['Daily_Return'].dropna()
        
        # Rolling statistics
        portfolio_data['Volatility_30d'] = returns.rolling(window=30).std() * np.sqrt(252)
        portfolio_data['Sharpe_30d'] = (returns.rolling(window=30).mean() * 252) / (returns.rolling(window=30).std() * np.sqrt(252))
        
        # Drawdown calculation
        rolling_max = portfolio_data['Portfolio_Value'].cummax()
        portfolio_data['Drawdown'] = (portfolio_data['Portfolio_Value'] - rolling_max) / rolling_max
        
        return portfolio_data
    
    @cache_data(cache_duration_hours=4)
    def get_market_indices(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get major market indices data
        
        Returns:
            Dictionary with market indices data
        """
        
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100', 
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market'
        }
        
        self.logger.info("📈 Fetching market indices data")
        
        return self.get_multiple_stocks(list(indices.keys()), period)
    
    def get_economic_indicators(self) -> Dict[str, float]:
        """
        Get key economic indicators (placeholder for FRED API integration)
        
        Returns:
            Dictionary with economic indicators
        """
        
        # Placeholder - will implement FRED API integration
        indicators = {
            'GDP_Growth': 2.1,
            'Unemployment_Rate': 3.7,
            'Inflation_Rate': 3.2,
            'Fed_Funds_Rate': 5.25,
            'VIX': 18.5
        }
        
        self.logger.info("📊 Economic indicators (mock data)")
        return indicators
    
    def save_to_cache(self, data: pd.DataFrame, filename: str) -> None:
        """Save data to cache"""
        filepath = self.cache_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"💾 Data cached to {filepath}")
    
    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        filepath = self.cache_dir / f"{filename}.pkl"
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"📂 Data loaded from cache: {filepath}")
            return data
        except FileNotFoundError:
            return None

# Create global market data instance
market_data = MarketDataProvider()

# Example usage and testing functions
def test_market_data():
    """Test the market data functionality"""
    
    print("🧪 Testing Market Data Provider...")
    
    # Test single stock
    try:
        data = market_data.get_stock_data('AAPL', period='1mo')
        print(f"✅ AAPL data: {len(data)} records")
        print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ AAPL test failed: {e}")
    
    # Test multiple stocks
    try:
        symbols = ['MSFT', 'GOOGL', 'TSLA']
        multi_data = market_data.get_multiple_stocks(symbols, period='1mo')
        print(f"✅ Multi-stock data: {len(multi_data)} symbols")
    except Exception as e:
        print(f"❌ Multi-stock test failed: {e}")
    
    # Test portfolio
    try:
        portfolio = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        stock_data, portfolio_returns = market_data.get_portfolio_data(portfolio, period='3mo')
        print(f"✅ Portfolio data: {len(portfolio_returns)} days")
        print(f"Portfolio return: {portfolio_returns['Cumulative_Return'].iloc[-1] - 1:.2%}")
    except Exception as e:
        print(f"❌ Portfolio test failed: {e}")

if __name__ == "__main__":
    test_market_data()