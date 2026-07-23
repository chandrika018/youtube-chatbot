import json
from typing import Any, Optional
from backend.app.config.config import settings

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        
        if settings.USE_REDIS:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    decode_responses=True
                )
                self.redis_client.ping()
                print("Successfully connected to Redis cache.")
            except Exception as e:
                print(f"Redis not available, falling back to local memory cache. Error: {e}")
                self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                print(f"Redis GET failed for key {key}: {e}")
        
        return self.local_cache.get(key)

    def set(self, key: str, value: Any, expire_seconds: int = 3600) -> bool:
        serialized = json.dumps(value)
        if self.redis_client:
            try:
                return self.redis_client.set(key, serialized, ex=expire_seconds)
            except Exception as e:
                print(f"Redis SET failed for key {key}: {e}")
        
        self.local_cache[key] = value
        return True

    def delete(self, key: str) -> bool:
        if self.redis_client:
            try:
                return bool(self.redis_client.delete(key))
            except Exception as e:
                print(f"Redis DELETE failed for key {key}: {e}")
        
        if key in self.local_cache:
            del self.local_cache[key]
            return True
        return False

    def clear(self):
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                print(f"Redis FLUSHDB failed: {e}")
        self.local_cache.clear()

cache = CacheManager()
