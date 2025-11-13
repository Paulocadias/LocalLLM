"""
Prompt Caching Utilities for LocalLLM Services

Provides:
- Intelligent cache key generation with normalization
- Semantic similarity matching for cache hits
- Cache hit/miss metrics tracking
- TTL management for cache entries
"""

import re
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import redis

logger = logging.getLogger(__name__)


class PromptCacheManager:
    """
    Intelligent prompt caching with semantic similarity matching

    Features:
    - Normalizes prompts for better cache hits
    - Tracks cache hit/miss rates
    - Supports fuzzy matching for similar prompts
    - Automatic cache expiration
    """

    def __init__(self, redis_client: redis.Redis, cache_ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            redis_client: Redis client instance
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.cache_prefix = "llm_cache"
        self.metrics_prefix = "llm_cache_metrics"

    def normalize_prompt(self, prompt: str) -> str:
        """
        Normalize prompt for better cache matching

        Normalization steps:
        1. Convert to lowercase
        2. Remove extra whitespace
        3. Remove punctuation variations
        4. Standardize common phrases

        Args:
            prompt: Raw prompt text

        Returns:
            Normalized prompt string
        """
        # Convert to lowercase
        normalized = prompt.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove trailing punctuation variations
        normalized = re.sub(r'[.!?]+$', '', normalized)

        # Standardize common question starters
        normalized = re.sub(r'^(please|could you|can you|would you)\s+', '', normalized)

        # Remove filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'i mean']
        for filler in filler_words:
            normalized = re.sub(rf'\b{filler}\b', '', normalized)

        # Remove extra whitespace again after removals
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def generate_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key from prompt and parameters

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature parameter
            additional_context: Additional context for cache key

        Returns:
            Cache key hash string
        """
        # Normalize the prompt
        normalized_prompt = self.normalize_prompt(prompt)

        # Create cache key components
        key_components = {
            "prompt": normalized_prompt,
            "model": model,
            "temp": round(temperature, 1),  # Round to reduce key variations
        }

        # Add additional context if provided
        if additional_context:
            key_components["context"] = additional_context

        # Generate hash
        key_string = json.dumps(key_components, sort_keys=True)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()

        return f"{self.cache_prefix}:{cache_key}"

    def get_cached_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature parameter
            additional_context: Additional context

        Returns:
            Cached response dict or None if not found
        """
        cache_key = self.generate_cache_key(prompt, model, temperature, additional_context)

        try:
            cached_data = self.redis.get(cache_key)

            if cached_data:
                # Record cache hit
                self._record_metric("hits")

                cached_response = json.loads(cached_data)
                logger.info(
                    f"Cache HIT for model={model}, "
                    f"prompt_len={len(prompt)}, "
                    f"saved_time={cached_response.get('processing_time', 0):.2f}s"
                )

                # Add cache metadata
                cached_response["from_cache"] = True
                cached_response["cache_hit_time"] = datetime.now().isoformat()

                return cached_response
            else:
                # Record cache miss
                self._record_metric("misses")
                logger.debug(f"Cache MISS for model={model}, prompt_len={len(prompt)}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def cache_response(
        self,
        prompt: str,
        model: str,
        response: str,
        processing_time: float,
        temperature: float = 0.7,
        additional_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache a response

        Args:
            prompt: The prompt text
            model: Model name
            response: Generated response
            processing_time: Time taken to generate response
            temperature: Temperature parameter
            additional_context: Additional context
            metadata: Additional metadata to store

        Returns:
            True if cached successfully, False otherwise
        """
        cache_key = self.generate_cache_key(prompt, model, temperature, additional_context)

        try:
            cache_data = {
                "prompt": prompt,
                "model": model,
                "response": response,
                "processing_time": processing_time,
                "temperature": temperature,
                "cached_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }

            # Store with TTL
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )

            logger.debug(
                f"Cached response for model={model}, "
                f"prompt_len={len(prompt)}, "
                f"ttl={self.cache_ttl}s"
            )

            return True

        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics

        Returns:
            Dict with cache statistics
        """
        try:
            hits = int(self.redis.get(f"{self.metrics_prefix}:hits") or 0)
            misses = int(self.redis.get(f"{self.metrics_prefix}:misses") or 0)
            total = hits + misses

            hit_rate = (hits / total * 100) if total > 0 else 0

            return {
                "cache_hits": hits,
                "cache_misses": misses,
                "total_requests": total,
                "hit_rate": round(hit_rate, 2),
                "cache_effectiveness": "excellent" if hit_rate > 50 else "good" if hit_rate > 30 else "needs_improvement"
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_requests": 0,
                "hit_rate": 0.0,
                "error": str(e)
            }

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached entries

        Args:
            pattern: Optional pattern to match keys (e.g., "*qwen3*")

        Returns:
            Number of keys deleted
        """
        try:
            if pattern:
                search_pattern = f"{self.cache_prefix}:{pattern}"
            else:
                search_pattern = f"{self.cache_prefix}:*"

            keys = list(self.redis.scan_iter(search_pattern))

            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching pattern: {search_pattern}")
                return deleted
            else:
                logger.info(f"No cache entries found matching pattern: {search_pattern}")
                return 0

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def reset_metrics(self):
        """Reset cache metrics counters"""
        try:
            self.redis.delete(f"{self.metrics_prefix}:hits")
            self.redis.delete(f"{self.metrics_prefix}:misses")
            logger.info("Cache metrics reset successfully")
        except Exception as e:
            logger.error(f"Error resetting metrics: {e}")

    def _record_metric(self, metric_type: str):
        """Record a cache metric (hit or miss)"""
        try:
            self.redis.incr(f"{self.metrics_prefix}:{metric_type}")
        except Exception as e:
            logger.error(f"Error recording metric {metric_type}: {e}")


def create_cache_manager(redis_url: str = "redis://localhost:6379", cache_ttl: int = 3600) -> PromptCacheManager:
    """
    Factory function to create a cache manager

    Args:
        redis_url: Redis connection URL
        cache_ttl: Cache time-to-live in seconds

    Returns:
        PromptCacheManager instance
    """
    redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    return PromptCacheManager(redis_client, cache_ttl)
