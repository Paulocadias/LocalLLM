"""
Shared authentication module for all Local LLM services
Provides unified API key validation with Redis-based storage, permissions, and rate limiting
"""
import hashlib
import json
import os
from datetime import datetime
from typing import Optional, List
from fastapi import Header, HTTPException
import redis
import logging

logger = logging.getLogger(__name__)


class APIKeyValidator:
    """
    Validates API keys against Redis storage with permission and rate limit checks
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize validator with Redis client

        Args:
            redis_client: Connected Redis client instance
        """
        self.redis = redis_client
        logger.info("APIKeyValidator initialized")

    async def verify_api_key(
        self,
        authorization: Optional[str] = Header(None),
        required_permissions: Optional[List[str]] = None
    ) -> dict:
        """
        Verify API key and check permissions

        Args:
            authorization: Authorization header (Bearer token)
            required_permissions: List of required permissions (e.g., ["chat", "admin"])

        Returns:
            dict: API key metadata

        Raises:
            HTTPException: 401 if missing/invalid header, 403 if invalid/expired/insufficient permissions, 429 if rate limited
        """
        # Check authorization header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid authorization header. Use: Authorization: Bearer <api_key>"
            )

        api_key = authorization.split(" ", 1)[1]
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Check Redis for key
        key_data_raw = self.redis.get(f"api_key:{key_hash}")
        if not key_data_raw:
            logger.warning(f"Invalid API key attempt: {key_hash[:8]}...")
            raise HTTPException(status_code=403, detail="Invalid API key")

        try:
            key_info = json.loads(key_data_raw)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse API key data for {key_hash[:8]}")
            raise HTTPException(status_code=500, detail="Internal authentication error")

        # Check if enabled
        if not key_info.get("enabled", False):
            logger.warning(f"Disabled API key used: {key_hash[:8]}...")
            raise HTTPException(status_code=403, detail="API key disabled")

        # Check expiration
        expires_at = key_info.get("expires_at")
        if expires_at:
            try:
                expires = datetime.fromisoformat(expires_at)
                if datetime.now() > expires:
                    logger.warning(f"Expired API key used: {key_hash[:8]}...")
                    raise HTTPException(status_code=403, detail="API key expired")
            except ValueError:
                logger.error(f"Invalid expiration date format for key {key_hash[:8]}")

        # Check permissions
        if required_permissions:
            key_permissions = set(key_info.get("permissions", []))

            # Check for wildcard permission
            if "*" not in key_permissions:
                required = set(required_permissions)
                if not required.issubset(key_permissions):
                    missing = required - key_permissions
                    logger.warning(
                        f"Insufficient permissions for key {key_hash[:8]}: missing {missing}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing permissions: {', '.join(missing)}"
                    )

        # Rate limiting check
        rate_limit = key_info.get("rate_limit")
        if rate_limit:
            await self._check_rate_limit(key_hash, rate_limit)

        logger.debug(f"API key validated: {key_info.get('name', 'Unknown')}")
        return key_info

    async def _check_rate_limit(self, key_hash: str, limits: dict):
        """
        Check and enforce rate limits

        Args:
            key_hash: SHA256 hash of API key
            limits: Rate limit configuration with requests_per_minute and requests_per_hour

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        minute_key = f"rate_limit:{key_hash}:minute"
        hour_key = f"rate_limit:{key_hash}:hour"

        # Increment counters
        minute_count = self.redis.incr(minute_key)
        hour_count = self.redis.incr(hour_key)

        # Set expiry on first increment
        if minute_count == 1:
            self.redis.expire(minute_key, 60)
        if hour_count == 1:
            self.redis.expire(hour_key, 3600)

        # Check limits
        max_per_minute = limits.get("requests_per_minute", 60)
        max_per_hour = limits.get("requests_per_hour", 1000)

        if minute_count > max_per_minute:
            logger.warning(
                f"Rate limit exceeded (per minute) for key {key_hash[:8]}: "
                f"{minute_count}/{max_per_minute}"
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {max_per_minute} requests per minute"
            )

        if hour_count > max_per_hour:
            logger.warning(
                f"Rate limit exceeded (per hour) for key {key_hash[:8]}: "
                f"{hour_count}/{max_per_hour}"
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {max_per_hour} requests per hour"
            )

    def migrate_legacy_api_key(self, legacy_key: str, name: str = "Legacy Admin Key"):
        """
        Migrate a legacy API key from environment variable to Redis

        Args:
            legacy_key: The API key to migrate
            name: Human-readable name for the key

        Returns:
            bool: True if migrated, False if already exists
        """
        key_hash = hashlib.sha256(legacy_key.encode()).hexdigest()

        # Check if already migrated
        if self.redis.exists(f"api_key:{key_hash}"):
            logger.info(f"API key {name} already migrated")
            return False

        # Create API key entry
        key_data = {
            "key_hash": key_hash,
            "name": name,
            "permissions": ["*"],  # Full permissions
            "rate_limit": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000
            },
            "enabled": True,
            "created_at": datetime.now().isoformat(),
            "migrated_from": "legacy_env"
        }

        self.redis.set(f"api_key:{key_hash}", json.dumps(key_data))
        logger.info(f"Migrated legacy API key: {name} ({key_hash[:8]}...)")
        return True


def create_validator_dependency(redis_client: redis.Redis, required_permissions: Optional[List[str]] = None):
    """
    Create a FastAPI dependency function for API key validation

    Args:
        redis_client: Connected Redis client
        required_permissions: List of required permissions

    Returns:
        Callable dependency function

    Example:
        verify_chat = create_validator_dependency(redis_client, ["chat"])

        @app.post("/chat")
        async def chat(auth: dict = Depends(verify_chat)):
            # Endpoint logic
    """
    validator = APIKeyValidator(redis_client)

    async def dependency(authorization: Optional[str] = Header(None)) -> dict:
        return await validator.verify_api_key(authorization, required_permissions)

    return dependency
