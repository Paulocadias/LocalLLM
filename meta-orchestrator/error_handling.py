"""
Error Handling Utilities for LocalLLM Services

Provides:
- Structured error responses
- Circuit breaker pattern
- Retry logic with exponential backoff
- Error categorization and logging
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from enum import Enum
from functools import wraps
from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for better handling"""
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    NOT_FOUND = "not_found"


class ErrorResponse(BaseModel):
    """Structured error response"""
    error: str
    category: ErrorCategory
    message: str
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None
    request_id: Optional[str] = None


class CircuitBreaker:
    """
    Circuit breaker pattern implementation

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(f"Circuit breaker entering HALF_OPEN state")
                self.state = "HALF_OPEN"
            else:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Service temporarily unavailable",
                        "category": ErrorCategory.SERVICE_UNAVAILABLE,
                        "retry_after": int(self.recovery_timeout - (time.time() - self.last_failure_time))
                    }
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        if self.state == "HALF_OPEN":
            logger.info(f"Circuit breaker recovered, entering CLOSED state")
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker threshold reached ({self.failure_count} failures), "
                f"entering OPEN state for {self.recovery_timeout}s"
            )
            self.state = "OPEN"


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )

                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def create_error_response(
    error: str,
    category: ErrorCategory,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    retry_after: Optional[int] = None,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Create a structured error response

    Args:
        error: Short error identifier
        category: Error category
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error details
        retry_after: Seconds to wait before retrying
        request_id: Request identifier for tracking

    Returns:
        HTTPException with structured error details
    """
    error_data = ErrorResponse(
        error=error,
        category=category,
        message=message,
        details=details,
        retry_after=retry_after,
        request_id=request_id
    )

    logger.error(
        f"Error response: {error} ({category}) - {message}",
        extra={
            "error": error,
            "category": category,
            "status_code": status_code,
            "details": details,
            "request_id": request_id
        }
    )

    return HTTPException(
        status_code=status_code,
        detail=error_data.dict(exclude_none=True)
    )


# Common error responses
def authentication_error(message: str = "Invalid or missing authentication credentials"):
    """Return authentication error"""
    return create_error_response(
        error="authentication_failed",
        category=ErrorCategory.AUTHENTICATION,
        message=message,
        status_code=401
    )


def rate_limit_error(retry_after: int = 60):
    """Return rate limit error"""
    return create_error_response(
        error="rate_limit_exceeded",
        category=ErrorCategory.RATE_LIMIT,
        message="Too many requests, please try again later",
        status_code=429,
        retry_after=retry_after
    )


def service_unavailable_error(service_name: str, retry_after: int = 30):
    """Return service unavailable error"""
    return create_error_response(
        error="service_unavailable",
        category=ErrorCategory.SERVICE_UNAVAILABLE,
        message=f"Service {service_name} is temporarily unavailable",
        status_code=503,
        details={"service": service_name},
        retry_after=retry_after
    )


def invalid_request_error(message: str, details: Optional[Dict[str, Any]] = None):
    """Return invalid request error"""
    return create_error_response(
        error="invalid_request",
        category=ErrorCategory.INVALID_REQUEST,
        message=message,
        status_code=400,
        details=details
    )


def model_error(model_name: str, error_details: str):
    """Return model error"""
    return create_error_response(
        error="model_error",
        category=ErrorCategory.MODEL_ERROR,
        message=f"Error processing request with model {model_name}",
        status_code=500,
        details={"model": model_name, "error": error_details}
    )


def timeout_error(operation: str, timeout_seconds: int):
    """Return timeout error"""
    return create_error_response(
        error="operation_timeout",
        category=ErrorCategory.TIMEOUT,
        message=f"Operation '{operation}' timed out after {timeout_seconds}s",
        status_code=504,
        details={"operation": operation, "timeout": timeout_seconds}
    )


def internal_error(message: str = "An internal error occurred", details: Optional[Dict[str, Any]] = None):
    """Return internal server error"""
    return create_error_response(
        error="internal_error",
        category=ErrorCategory.INTERNAL_ERROR,
        message=message,
        status_code=500,
        details=details
    )


def not_found_error(resource: str, resource_id: Optional[str] = None):
    """Return not found error"""
    return create_error_response(
        error="not_found",
        category=ErrorCategory.NOT_FOUND,
        message=f"Resource '{resource}' not found",
        status_code=404,
        details={"resource": resource, "id": resource_id} if resource_id else {"resource": resource}
    )
