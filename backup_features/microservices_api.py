from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import aioredis
import json
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

# Pydantic models for API contracts
class PortfolioRequest(BaseModel):
    """Request model for portfolio optimization."""
    assets: List[str] = Field(..., description="List of asset symbols")
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0, description="Risk tolerance (0-1)")
    investment_amount: float = Field(..., gt=0, description="Investment amount in USD")
    optimization_method: str = Field("markowitz", description="Optimization algorithm")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Additional constraints")

class PortfolioResponse(BaseModel):
    """Response model for portfolio optimization."""
    portfolio_id: str
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_timestamp: datetime
    status: str

class RiskMetrics(BaseModel):
    """Risk metrics response model."""
    var_95: float = Field(..., description="95% Value at Risk")
    cvar_95: float = Field(..., description="95% Conditional VaR")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    beta: float = Field(..., description="Portfolio beta")
    volatility: float = Field(..., description="Annualized volatility")

# Global Redis connection pool
redis_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    global redis_pool
    # Startup
    redis_pool = aioredis.ConnectionPool.from_url(
        "redis://localhost:6379", max_connections=20
    )
    yield
    # Shutdown
    await redis_pool.disconnect()

# Initialize FastAPI app
app = FastAPI(
    title="AI Portfolio Manager API",
    description="Enterprise-grade portfolio optimization microservice",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def get_redis_connection():
    """Get Redis connection from pool."""
    return aioredis.Redis(connection_pool=redis_pool)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (simplified for demonstration)."""
    token = credentials.credentials
    # In production, verify JWT signature, expiration, etc.
    if token != "demo_token_123":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

# Cache decorator for expensive operations
def cache_result(expiration: int = 300):
    """Decorator to cache API responses in Redis."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"cache:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            redis = await get_redis_connection()
            
            # Try to get cached result
            cached_result = await redis.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await redis.setex(cache_key, expiration, json.dumps(result, default=str))
            
            return result
        return wrapper
    return decorator

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AI Portfolio Manager API", "status": "healthy", "version": "2.1.0"}

@app.post("/api/v1/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(
    request: PortfolioRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Optimize portfolio allocation based on request parameters.
    Demonstrates async processing with background tasks.
    """
    try:
        # Generate unique portfolio ID
        portfolio_id = f"portfolio_{int(datetime.utcnow().timestamp())}"
        
        # Simulate portfolio optimization (replace with actual optimization)
        weights = await simulate_portfolio_optimization(request)
        
        # Calculate metrics (simplified)
        expected_return = sum(w * 0.08 for w in weights.values())  # Simplified
        expected_risk = sum(w * 0.15 for w in weights.values()) * 0.5  # Simplified
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        
        response = PortfolioResponse(
            portfolio_id=portfolio_id,
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            optimization_timestamp=datetime.utcnow(),
            status="completed"
        )
        
        # Store result in cache for later retrieval
        background_tasks.add_task(store_portfolio_result, portfolio_id, response)
        
        return response
        
    except Exception as e:
        logging.error(f"Portfolio optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Portfolio optimization failed")

@app.get("/api/v1/portfolio/{portfolio_id}/risk", response_model=RiskMetrics)
@cache_result(expiration=600)  # Cache for 10 minutes
async def calculate_risk_metrics(
    portfolio_id: str,
    token: str = Depends(verify_token)
):
    """
    Calculate comprehensive risk metrics for a portfolio.
    Demonstrates caching for expensive computations.
    """
    try:
        # Retrieve portfolio from cache
        portfolio = await get_portfolio_from_cache(portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Simulate risk calculation (replace with actual risk engine)
        risk_metrics = await simulate_risk_calculation(portfolio)
        
        return risk_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Risk calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Risk calculation failed")

@app.get("/api/v1/market/sentiment")
async def get_market_sentiment(
    symbols: Optional[List[str]] = None,
    token: str = Depends(verify_token)
):
    """
    Get real-time market sentiment analysis.
    Demonstrates integration with external data sources.
    """
    try:
        # Simulate sentiment analysis (replace with actual sentiment engine)
        sentiment_data = await simulate_sentiment_analysis(symbols or ["SPY", "AAPL", "MSFT"])
        
        return {
            "timestamp": datetime.utcnow(),
            "sentiment_scores": sentiment_data,
            "overall_sentiment": "bullish" if sum(sentiment_data.values()) > 0 else "bearish"
        }
        
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

@app.post("/api/v1/backtest")
async def run_backtest(
    portfolio_weights: Dict[str, float],
    start_date: datetime,
    end_date: datetime,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Run portfolio backtest asynchronously.
    Demonstrates long-running background tasks.
    """
    try:
        # Generate backtest job ID
        job_id = f"backtest_{int(datetime.utcnow().timestamp())}"
        
        # Start backtest in background
        background_tasks.add_task(
            run_backtest_async, job_id, portfolio_weights, start_date, end_date
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Backtest started. Use /api/v1/backtest/{job_id}/status to check progress."
        }
        
    except Exception as e:
        logging.error(f"Backtest initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Backtest initialization failed")

@app.get("/api/v1/backtest/{job_id}/status")
async def get_backtest_status(job_id: str, token: str = Depends(verify_token)):
    """Get backtest job status and results."""
    redis = await get_redis_connection()
    status_key = f"backtest_status:{job_id}"
    result_key = f"backtest_result:{job_id}"
    
    status = await redis.get(status_key)
    if not status:
        raise HTTPException(status_code=404, detail="Backtest job not found")
    
    response = {"job_id": job_id, "status": status.decode()}
    
    if status.decode() == "completed":
        result = await redis.get(result_key)
        if result:
            response["results"] = json.loads(result)
    
    return response

# Helper functions for simulation (replace with actual implementations)

async def simulate_portfolio_optimization(request: PortfolioRequest) -> Dict[str, float]:
    """Simulate portfolio optimization process."""
    await asyncio.sleep(0.1)  # Simulate processing time
    
    # Simple equal weighting with some randomization
    n_assets = len(request.assets)
    base_weight = 1.0 / n_assets
    
    weights = {}
    for asset in request.assets:
        # Add some random variation based on risk tolerance
        variation = (0.5 - request.risk_tolerance) * 0.2
        weight = base_weight + variation * (hash(asset) % 100 / 500 - 0.1)
        weights[asset] = max(0.01, min(0.5, weight))  # Constrain weights
    
    # Normalize to sum to 1
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights

async def simulate_risk_calculation(portfolio: Dict) -> RiskMetrics:
    """Simulate risk metrics calculation."""
    await asyncio.sleep(0.2)  # Simulate processing time
    
    return RiskMetrics(
        var_95=0.045,
        cvar_95=0.067,
        max_drawdown=0.12,
        beta=0.95,
        volatility=0.16
    )

async def simulate_sentiment_analysis(symbols: List[str]) -> Dict[str, float]:
    """Simulate sentiment analysis."""
    await asyncio.sleep(0.1)  # Simulate processing time
    
    return {symbol: (hash(symbol) % 200 - 100) / 100 for symbol in symbols}

async def store_portfolio_result(portfolio_id: str, result: PortfolioResponse):
    """Store portfolio result in Redis cache."""
    redis = await get_redis_connection()
    cache_key = f"portfolio:{portfolio_id}"
    await redis.setex(cache_key, 3600, result.model_dump_json())  # Cache for 1 hour

async def get_portfolio_from_cache(portfolio_id: str) -> Optional[Dict]:
    """Retrieve portfolio from Redis cache."""
    redis = await get_redis_connection()
    cache_key = f"portfolio:{portfolio_id}"
    result = await redis.get(cache_key)
    return json.loads(result) if result else None

async def run_backtest_async(job_id: str, weights: Dict[str, float], 
                           start_date: datetime, end_date: datetime):
    """Run backtest in background task."""
    redis = await get_redis_connection()
    status_key = f"backtest_status:{job_id}"
    result_key = f"backtest_result:{job_id}"
    
    try:
        await redis.setex(status_key, 3600, "running")
        
        # Simulate backtest processing
        await asyncio.sleep(5)  # Simulate long computation
        
        # Generate mock results
        results = {
            "total_return": 0.124,
            "sharpe_ratio": 1.18,
            "max_drawdown": 0.082,
            "volatility": 0.145,
            "trades": 45,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        await redis.setex(status_key, 3600, "completed")
        await redis.setex(result_key, 3600, json.dumps(results, default=str))
        
    except Exception as e:
        await redis.setex(status_key, 3600, "failed")
        logging.error(f"Backtest {job_id} failed: {str(e)}")
