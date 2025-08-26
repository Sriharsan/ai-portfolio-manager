# API Documentation

## Core Modules API Reference

### Data Loader (`src/data/data_loader.py`)

#### `get_portfolio_analysis(portfolio_weights, period='1y')`
Get comprehensive portfolio analysis.

**Parameters:**
- `portfolio_weights` (Dict[str, float]): Asset allocation (must sum to 1.0)
- `period` (str): Time period ('1mo', '6mo', '1y', '2y')

**Returns:**
- Dict containing performance metrics, risk assessment, and AI insights

**Example:**
```python
from src.data.data_loader import data_loader

portfolio = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
analysis = data_loader.get_portfolio_analysis(portfolio, '6mo')
```

#### `get_stock_analysis(symbol, period='6mo')`
Get individual stock analysis.

**Parameters:**
- `symbol` (str): Stock ticker symbol
- `period` (str): Analysis period

**Returns:**
- Dict with stock data, performance metrics, and AI insights

### Market Data (`src/data/market_data.py`)

#### `get_stock_data(symbol, period='1y', interval='1d')`
Fetch stock price data with technical indicators.

**Parameters:**
- `symbol` (str): Stock ticker
- `period` (str): Data period
- `interval` (str): Data frequency

**Returns:**
- DataFrame with OHLCV data and technical indicators

### Portfolio Optimizer (`src/models/portfolio_optimizer.py`)

#### `optimize_portfolio(returns, method='max_sharpe')`
Optimize portfolio allocation using Modern Portfolio Theory.

**Parameters:**
- `returns` (DataFrame): Asset return data
- `method` (str): Optimization method ('max_sharpe', 'min_volatility')

**Returns:**
- Dict with optimal weights and performance metrics

### LLM Engine (`src/models/llm_engine.py`)

#### `analyze_sentiment(text)`
Analyze financial text sentiment.

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
- Dict with sentiment label, score, and confidence

#### `generate_market_insight(stock_data, symbol)`
Generate AI-powered market insights.

**Parameters:**
- `stock_data` (DataFrame): Historical price data
- `symbol` (str): Stock symbol

**Returns:**
- String with market analysis and recommendations

## Visualization API

### Charts (`src/visualization/charts.py`)

#### `create_performance_chart(data, title)`
Create interactive performance chart.

#### `create_allocation_pie(weights)`
Create portfolio allocation pie chart.

#### `create_risk_metrics_bar(metrics)`
Create risk metrics visualization.

### Dashboards (`src/visualization/dashboards.py`)

#### `portfolio_comparison_chart(portfolios)`
Compare multiple portfolio performances.

#### `risk_return_scatter(data)`
Create risk-return scatter plot.

## Error Handling

All functions handle errors gracefully and return error information in result dictionaries:

```python
result = data_loader.get_portfolio_analysis(portfolio)
if 'error' in result:
    print(f"Analysis failed: {result['error']}")
else:
    # Process successful result
    metrics = result['performance_metrics']
```

## Rate Limits

- **Alpha Vantage**: 5 calls/minute (free tier)
- **yfinance**: No official limits, use responsibly
- **FRED**: 120 calls/minute

## Configuration

All API keys and settings are managed through the `config.py` file and environment variables.