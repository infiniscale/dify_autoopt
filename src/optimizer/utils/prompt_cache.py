"""
Prompt Cache for LLM API Responses.

Date: 2025-11-18
Author: backend-developer
Description: Thread-safe in-memory cache with TTL support for prompt optimization results.
"""

import hashlib
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class PromptCache:
    """Thread-safe in-memory cache for prompt optimization results.

    This cache stores LLM responses to reduce API calls and costs:
    - TTL-based expiration (default 24 hours)
    - Maximum size limit with LRU eviction
    - MD5-based cache keys
    - Thread-safe operations
    - Cache statistics (hit rate, size)

    Example:
        >>> cache = PromptCache(ttl_seconds=3600, max_size=1000)
        >>> cache.set("My prompt", "Optimized result", "llm_clarity")
        >>> result = cache.get("My prompt", "llm_clarity")
        >>> print(result)  # "Optimized result"
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, ttl_seconds: int = 86400, max_size: int = 1000):
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 24 hours)
            max_size: Maximum number of cache entries (default: 1000)

        Example:
            >>> cache = PromptCache(ttl_seconds=3600, max_size=500)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _generate_key(self, prompt: str, strategy: str = "") -> str:
        """Generate cache key using MD5 hash.

        Args:
            prompt: Prompt text
            strategy: Optimization strategy

        Returns:
            MD5 hash as hexadecimal string

        Example:
            >>> key = cache._generate_key("My prompt", "llm_clarity")
            >>> print(len(key))  # 32 (MD5 hex length)
        """
        content = f"{prompt}:{strategy}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, prompt: str, strategy: str = "") -> Optional[str]:
        """Get cached optimization result.

        Args:
            prompt: Prompt text to look up
            strategy: Optimization strategy used

        Returns:
            Cached result if found and not expired, None otherwise

        Example:
            >>> result = cache.get("My prompt", "llm_clarity")
            >>> if result:
            ...     print("Cache hit!")
        """
        key = self._generate_key(prompt, strategy)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if datetime.now() > entry["expires_at"]:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                return None

            # Update access time for LRU
            entry["last_accessed"] = datetime.now()
            self._hits += 1
            return entry["result"]

    def set(self, prompt: str, result: str, strategy: str = "") -> None:
        """Set cache entry with TTL.

        Args:
            prompt: Prompt text (cache key component)
            result: Optimization result to cache
            strategy: Optimization strategy (cache key component)

        Example:
            >>> cache.set("My prompt", "Optimized version", "llm_clarity")
        """
        key = self._generate_key(prompt, strategy)

        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            # Store entry
            now = datetime.now()
            self._cache[key] = {
                "result": result,
                "created_at": now,
                "expires_at": now + timedelta(seconds=self._ttl),
                "last_accessed": now,
                "prompt_length": len(prompt),
                "strategy": strategy
            }

    def _evict_lru(self) -> None:
        """Evict least recently used entry (LRU policy).

        This method is called when cache is full and a new entry needs to be added.
        It removes the entry with the oldest last_accessed timestamp.
        """
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["last_accessed"]
        )

        # Remove it
        del self._cache[lru_key]
        self._evictions += 1

    def clear_expired(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed

        Example:
            >>> removed = cache.clear_expired()
            >>> print(f"Cleared {removed} expired entries")
        """
        now = datetime.now()
        removed = 0

        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry["expires_at"]
            ]

            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - size: Current number of entries
            - max_size: Maximum cache size
            - evictions: Number of evicted entries

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
            >>> print(f"Cache size: {stats['size']}/{stats['max_size']}")
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self._max_size,
                "evictions": self._evictions
            }

    def clear(self) -> None:
        """Clear all cache entries and reset statistics.

        Example:
            >>> cache.clear()
            >>> assert cache.get_stats()["size"] == 0
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_size_bytes(self) -> int:
        """Estimate cache size in bytes.

        Returns:
            Approximate memory usage in bytes

        Note:
            This is an estimate based on string lengths and metadata size.

        Example:
            >>> size_kb = cache.get_size_bytes() / 1024
            >>> print(f"Cache size: {size_kb:.2f} KB")
        """
        with self._lock:
            total_bytes = 0

            for entry in self._cache.values():
                # Estimate: result string + metadata
                total_bytes += len(entry["result"]) * 2  # Unicode chars (2 bytes each)
                total_bytes += entry["prompt_length"] * 2
                total_bytes += 100  # Metadata overhead (timestamps, etc.)

            return total_bytes
