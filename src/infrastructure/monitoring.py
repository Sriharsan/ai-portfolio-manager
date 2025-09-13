import time
import logging
from functools import wraps
from typing import Dict, Callable
from datetime import datetime

class PerformanceMonitor:
    """Monitor application performance and health"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.record_metric(f"{func.__name__}_duration", execution_time)
                return result
            except Exception as e:
                self.record_metric(f"{func.__name__}_errors", 1)
                raise e
        return wrapper
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now()
        })
        
        # Keep only last 1000 records
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def get_health_status(self) -> Dict:
        """Get application health status"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now(),
            'metrics_count': len(self.metrics),
            'uptime': time.time()
        }

# Global monitor instance
performance_monitor = PerformanceMonitor()
