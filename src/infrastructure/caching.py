import redis
import pickle
import logging
from typing import Any, Optional
from datetime import timedelta

class CacheManager:
    """Redis-based caching for performance optimization"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            self.logger = logging.getLogger(__name__)
            self.enabled = True
        except:
            self.redis_client = None
            self.enabled = False
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Redis not available, caching disabled")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.enabled:
            return None
        
        try:
            data = self.redis_client.get(key)
            return pickle.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Cache get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cached value with TTL"""
        if not self.enabled:
            return False
        
        try:
            serialized = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            self.logger.error(f"Cache set failed: {e}")
            return False

# Global cache instance
cache_manager = CacheManager()