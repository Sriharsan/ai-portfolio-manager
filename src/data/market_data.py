# src/data/market_data.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from fredapi import Fred

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  BlackRock-level market data provider with multi-source integration

class InstitutionalMarketDataProvider:
    
    def __init__(self):
       
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from config.config import app_config
            self.alpha_vantage_key = app_config.ALPHA_VANTAGE_API_KEY
            self.fred_key = app_config.FRED_API_KEY
            self.cache_dir = app_config.CACHE_DIR
        except ImportError:
            import os
            self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
            self.fred_key = os.getenv('FRED_API_KEY', '')
            self.cache_dir = Path('data/cache')
        
        # Initialize API clients
        self._setup_api_clients()
        
        # Expanded asset universe
        self.asset_universe = self._load_institutional_universe()
        
        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 0.2  # 5 calls per second max
        
        logger.info("Institutional Market Data Provider initialized")
        logger.info(f"Available assets: {len(self.asset_universe)} symbols")
    
    def _setup_api_clients(self):
        """Initialize premium API clients"""
        try:
            if self.alpha_vantage_key:
                self.av_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
                self.av_fundamentals = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
                logger.info("✅ Alpha Vantage API initialized")
            else:
                self.av_ts = None
                self.av_fundamentals = None
                logger.warning("⚠️ Alpha Vantage API key missing")
            
            if self.fred_key:
                self.fred = Fred(api_key=self.fred_key)
                logger.info("✅ FRED API initialized")
            else:
                self.fred = None
                logger.warning("⚠️ FRED API key missing")
                
        except Exception as e:
            logger.error(f"API client setup failed: {e}")
            self.av_ts = None
            self.av_fundamentals = None
            self.fred = None
    
    def _load_institutional_universe(self) -> Dict[str, Dict]:
        
        universe = {
            # Large Cap US Stocks
            'AAPL': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'MSFT': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'GOOGL': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'AMZN': {'sector': 'Consumer Discretionary', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'TSLA': {'sector': 'Consumer Discretionary', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'NVDA': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'META': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'BRK-B': {'sector': 'Financials', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'JPM': {'sector': 'Financials', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'JNJ': {'sector': 'Healthcare', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'V': {'sector': 'Financials', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'PG': {'sector': 'Consumer Staples', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'UNH': {'sector': 'Healthcare', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'HD': {'sector': 'Consumer Discretionary', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            'MA': {'sector': 'Financials', 'type': 'equity', 'region': 'US', 'cap': 'large'},
            
            # Mid Cap Stocks
            'AMD': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'mid'},
            'INTC': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'mid'},
            'NFLX': {'sector': 'Communication Services', 'type': 'equity', 'region': 'US', 'cap': 'mid'},
            'CRM': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'mid'},
            'ADBE': {'sector': 'Technology', 'type': 'equity', 'region': 'US', 'cap': 'mid'},
            
            # International Exposure
            'VEA': {'sector': 'International', 'type': 'etf', 'region': 'Developed', 'cap': 'mixed'},
            'VWO': {'sector': 'International', 'type': 'etf', 'region': 'Emerging', 'cap': 'mixed'},
            'EFA': {'sector': 'International', 'type': 'etf', 'region': 'EAFE', 'cap': 'mixed'},
            
            # Fixed Income
            'TLT': {'sector': 'Government Bonds', 'type': 'etf', 'duration': 'long', 'credit': 'AAA'},
            'IEF': {'sector': 'Government Bonds', 'type': 'etf', 'duration': 'intermediate', 'credit': 'AAA'},
            'SHY': {'sector': 'Government Bonds', 'type': 'etf', 'duration': 'short', 'credit': 'AAA'},
            'LQD': {'sector': 'Corporate Bonds', 'type': 'etf', 'credit': 'investment_grade'},
            'HYG': {'sector': 'Corporate Bonds', 'type': 'etf', 'credit': 'high_yield'},
            'TIP': {'sector': 'TIPS', 'type': 'etf', 'inflation_protected': True},
            
            # Sector ETFs
            'XLK': {'sector': 'Technology', 'type': 'etf', 'region': 'US'},
            'XLF': {'sector': 'Financials', 'type': 'etf', 'region': 'US'},
            'XLE': {'sector': 'Energy', 'type': 'etf', 'region': 'US'},
            'XLV': {'sector': 'Healthcare', 'type': 'etf', 'region': 'US'},
            'XLP': {'sector': 'Consumer Staples', 'type': 'etf', 'region': 'US'},
            'XLY': {'sector': 'Consumer Discretionary', 'type': 'etf', 'region': 'US'},
            'XLI': {'sector': 'Industrials', 'type': 'etf', 'region': 'US'},
            'XLU': {'sector': 'Utilities', 'type': 'etf', 'region': 'US'},
            'XLRE': {'sector': 'Real Estate', 'type': 'etf', 'region': 'US'},
            'XLB': {'sector': 'Materials', 'type': 'etf', 'region': 'US'},
            
            # Commodities and Alternatives
            'GLD': {'sector': 'Commodities', 'type': 'etf', 'commodity': 'gold'},
            'SLV': {'sector': 'Commodities', 'type': 'etf', 'commodity': 'silver'},
            'USO': {'sector': 'Commodities', 'type': 'etf', 'commodity': 'oil'},
            'VNQ': {'sector': 'Real Estate', 'type': 'etf', 'region': 'US'},
            'IYR': {'sector': 'Real Estate', 'type': 'etf', 'region': 'US'},
            
            # Market Benchmarks
            'SPY': {'sector': 'Market', 'type': 'etf', 'benchmark': 'S&P 500'},
            'QQQ': {'sector': 'Market', 'type': 'etf', 'benchmark': 'NASDAQ 100'},
            'IWM': {'sector': 'Market', 'type': 'etf', 'benchmark': 'Russell 2000'},
            'VTI': {'sector': 'Market', 'type': 'etf', 'benchmark': 'Total Stock Market'},
            'VXUS': {'sector': 'Market', 'type': 'etf', 'benchmark': 'International Stock Market'},
            
            # Volatility and Alternative Strategies
            'VIX': {'sector': 'Volatility', 'type': 'index', 'asset_class': 'volatility'},
            'VIXY': {'sector': 'Volatility', 'type': 'etf', 'strategy': 'volatility'},
            
            # Currency ETFs
            'UUP': {'sector': 'Currency', 'type': 'etf', 'currency': 'USD'},
            'FXE': {'sector': 'Currency', 'type': 'etf', 'currency': 'EUR'},
        }
        
        return universe
    
    def get_available_assets(self, filter_by: Optional[Dict] = None) -> List[str]:
        
        if not filter_by:
            return list(self.asset_universe.keys())
        
        filtered = []
        for symbol, attributes in self.asset_universe.items():
            match = True
            for key, value in filter_by.items():
                if attributes.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(symbol)
        
        return filtered
    
    def get_stock_data_premium(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        
        self._rate_limit()
        
        # Try Alpha Vantage first for premium data
        if self.av_ts and symbol in self.asset_universe:
            try:
                logger.info(f"Fetching {symbol} from Alpha Vantage")
                
                if period in ['1mo', '3mo']:
                    # Use intraday for short periods
                    data, meta_data = self.av_ts.get_intraday(
                        symbol=symbol, interval='60min', outputsize='full'
                    )
                    data = data.resample('D').agg({
                        '1. open': 'first',
                        '2. high': 'max',
                        '3. low': 'min',
                        '4. close': 'last',
                        '5. volume': 'sum'
                    }).dropna()
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                else:
                    # Use daily for longer periods
                    data, meta_data = self.av_ts.get_daily_adjusted(
                        symbol=symbol, outputsize='full'
                    )
                    data = data[['1. open', '2. high', '3. low', '4. close', '6. volume']]
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Filter by period
                data = self._filter_by_period(data, period)
                
                if not data.empty:
                    data = self._add_technical_indicators(data)
                    logger.info(f"✅ Alpha Vantage: {len(data)} records for {symbol}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
        
        # Fallback to yfinance
        return self._get_yfinance_data(symbol, period, interval)
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        
        return self.get_stock_data_premium(symbol, period, interval)
    
    def _get_yfinance_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        
        try:
            logger.info(f"Fetching {symbol} from yfinance")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if data.empty:
                logger.warning(f"No yfinance data for {symbol}")
                return self._create_realistic_mock_data(symbol, period)
            
            # Clean and enhance
            data = self._clean_stock_data(data)
            data = self._add_technical_indicators(data)
            
            logger.info(f"✅ yfinance: {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"yfinance failed for {symbol}: {e}")
            return self._create_realistic_mock_data(symbol, period)
    
    def _filter_by_period(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        
        if data.empty:
            return data
        
        end_date = datetime.now()
        
        if period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '2y':
            start_date = end_date - timedelta(days=730)
        else:
            return data
        
        return data[data.index >= start_date]
    
    def get_economic_indicators(self) -> Dict[str, float]:
        
        indicators = {}
        
        if not self.fred:
            return indicators
        
        try:
            # Key economic indicators
            fred_series = {
                'gdp_growth': 'GDPC1',
                'inflation_rate': 'CPIAUCSL',
                'unemployment_rate': 'UNRATE',
                'fed_funds_rate': 'FEDFUNDS',
                '10y_treasury': 'GS10',
                '2y_treasury': 'GS2',
                'consumer_confidence': 'UMCSENT',
                'housing_starts': 'HOUST'
            }
            
            for name, series_id in fred_series.items():
                try:
                    data = self.fred.get_series(series_id, limit=1)
                    if not data.empty:
                        indicators[name] = float(data.iloc[-1])
                except:
                    continue
            
            logger.info(f"Retrieved {len(indicators)} economic indicators")
            
        except Exception as e:
            logger.error(f"FRED data retrieval failed: {e}")
        
        return indicators
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        
        sector_etfs = {
            'Technology': 'XLK',
            'Financials': 'XLF',
            'Healthcare': 'XLV',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        sector_performance = {}
        
        for sector, etf in sector_etfs.items():
            try:
                data = self.get_stock_data_premium(etf, '1mo')
                if not data.empty and 'Daily_Return' in data.columns:
                    sector_performance[sector] = {
                        'symbol': etf,
                        'current_price': data['Close'].iloc[-1],
                        'monthly_return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100,
                        'volatility': data['Daily_Return'].std() * np.sqrt(252) * 100
                    }
            except Exception as e:
                logger.error(f"Sector performance failed for {sector}: {e}")
        
        return sector_performance
    
    def _create_realistic_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        
        logger.info(f"Creating realistic mock data for {symbol}")
        
        # Get asset characteristics
        asset_info = self.asset_universe.get(symbol, {})
        asset_type = asset_info.get('type', 'equity')
        sector = asset_info.get('sector', 'Mixed')
        
        # Date range
        end_date = datetime.now()
        if period == '1mo':
            start_date = end_date - timedelta(days=30)
            n_days = 22  # Business days
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
            n_days = 65
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
            n_days = 126
        else:
            start_date = end_date - timedelta(days=365)
            n_days = 252
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
        
        # Set parameters based on asset type
        if asset_type == 'etf' and 'Bonds' in sector:
            base_price = 100.0
            daily_vol = 0.003
            drift = 0.0002
        elif asset_type == 'etf' and sector == 'Commodities':
            base_price = 150.0
            daily_vol = 0.02
            drift = 0.0001
        elif sector == 'Technology':
            base_price = np.random.uniform(150, 300)
            daily_vol = 0.025
            drift = 0.001
        else:
            base_price = np.random.uniform(50, 200)
            daily_vol = 0.018
            drift = 0.0008
        
        # Generate realistic price series
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(drift, daily_vol, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            intraday_vol = daily_vol * np.random.uniform(0.3, 1.5)
            high = close * (1 + intraday_vol/2)
            low = close * (1 - intraday_vol/2)
            open_price = close + np.random.normal(0, close * daily_vol/4)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Volume based on asset type
            if asset_type == 'etf':
                base_volume = np.random.randint(10000000, 50000000)
            else:
                base_volume = np.random.randint(1000000, 20000000)
            
            volume = int(base_volume * np.random.uniform(0.5, 2.0))
            
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
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        
        if data.empty:
            return data
        
        # Daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Moving averages
        if len(data) >= 5:
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
        if len(data) >= 20:
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean() if len(data) >= 50 else data['SMA_20']
        if len(data) >= 200:
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential moving averages
        if len(data) >= 12:
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
        if len(data) >= 26:
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        if len(data) >= 26:
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
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
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / sma20
        
        # Volatility measures
        if len(data) >= 20:
            data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        return data
    
    def _clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Remove rows with missing essential data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Fix price relationships
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        # Remove extreme outliers (price changes > 50%)
        returns = data['Close'].pct_change()
        outlier_mask = (returns.abs() < 0.5) | returns.isna()
        data = data[outlier_mask].copy()
        
        # Ensure positive prices and volumes
        data = data[data['Close'] > 0]
        data = data[data['Volume'] > 0]
        
        return data
    
    def _rate_limit(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        
        self.last_call_time = time.time()
        
    def get_portfolio_data(self, portfolio_weights: Dict[str, float], period: str = '1y') -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
       
        stock_data = {}
        portfolio_returns_data = []
    
        # Get data for each symbol
        for symbol in portfolio_weights.keys():
            try:
                data = self.get_stock_data_premium(symbol, period)
                if not data.empty:
                    stock_data[symbol] = data
                    logger.info(f"Retrieved {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                continue
    
        # Calculate portfolio returns if we have stock data
        if stock_data:
            try:
                portfolio_returns = self._calculate_portfolio_returns(stock_data, portfolio_weights)
                return stock_data, portfolio_returns
            except Exception as e:
                logger.error(f"Portfolio calculation failed: {e}")
                return stock_data, pd.DataFrame()

        return {}, pd.DataFrame()

    def _calculate_portfolio_returns(self, stock_data: Dict[str, pd.DataFrame], 
                                   weights: Dict[str, float]) -> pd.DataFrame:

        # Get all dates that appear in any stock
        all_dates = set()
        for data in stock_data.values():
            all_dates.update(data.index)

        all_dates = sorted(list(all_dates))

        if not all_dates:
            return pd.DataFrame()

        # Create portfolio returns dataframe
        portfolio_data = pd.DataFrame(index=all_dates)

        # Calculate daily portfolio returns
        daily_returns = []
        portfolio_values = []

        for date in all_dates:
            daily_return = 0
            total_weight = 0

            for symbol, weight in weights.items():
                if symbol in stock_data:
                    symbol_data = stock_data[symbol]
                    if date in symbol_data.index and 'Daily_Return' in symbol_data.columns:
                        symbol_return = symbol_data.loc[date, 'Daily_Return']
                        if not pd.isna(symbol_return):
                            daily_return += weight * symbol_return
                            total_weight += weight

            # Normalize by actual weight (in case some stocks missing data)
            if total_weight > 0:
                daily_return = daily_return / total_weight

            daily_returns.append(daily_return)

        # Add to dataframe
        portfolio_data['Daily_Return'] = daily_returns
        portfolio_data['Cumulative_Return'] = (1 + portfolio_data['Daily_Return']).cumprod() - 1

        # Add portfolio value (assuming $100k starting value)
        starting_value = 100000
        portfolio_data['Portfolio_Value'] = starting_value * (1 + portfolio_data['Cumulative_Return'])

        # Add rolling volatility
        if len(portfolio_data) >= 30:
            portfolio_data['Volatility_30d'] = portfolio_data['Daily_Return'].rolling(30).std() * np.sqrt(252)

        # Add drawdown
        running_max = portfolio_data['Cumulative_Return'].cummax()
        portfolio_data['Drawdown'] = (portfolio_data['Cumulative_Return'] - running_max) / (1 + running_max)

        return portfolio_data.dropna()

# Global instance
market_data_provider = InstitutionalMarketDataProvider()