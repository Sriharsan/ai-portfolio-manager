"""
Utility Decorators for Performance and Error Handling
"""

import time
import functools
import logging
from typing import Callable, Any

def timing_decorator(func: Callable) -> Callable:
    """Measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} executed in {end-start:.2f}s")
        return result
    return wrapper

def error_handler(default_return=None):
    """Handle exceptions gracefully"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"{func.__name__} failed: {e}")
                return default_return
        return wrapper
    return decorator

def cache_result(expiry_seconds: int = 3600):
    """Simple result caching"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < expiry_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        return wrapper
    return decorator

def validate_portfolio_input(func: Callable) -> Callable:
    """Validate portfolio input parameters"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Check if portfolio weights are provided and valid
        if 'portfolio' in kwargs or (len(args) > 0 and isinstance(args[0], dict)):
            portfolio = kwargs.get('portfolio', args[0] if args else {})
            
            if not isinstance(portfolio, dict):
                raise ValueError("Portfolio must be a dictionary")
            
            total_weight = sum(portfolio.values())
            if abs(total_weight - 1.0) > 0.01:
                logging.warning(f"Portfolio weights sum to {total_weight}, not 1.0")
        
        return func(*args, **kwargs)
    return wrapper