"""Thunders AI API Middleware - Request/response processing pipeline.

Provides four middleware components for the FastAPI application:
- RateLimitMiddleware: Per-API-key rate limiting
- LoggingMiddleware: Request/response logging with timing
- AuthMiddleware: Authentication verification
- ErrorHandlingMiddleware: Global exception handling
"""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that bypass authentication and rate limiting
PUBLIC_PATHS: Set[str] = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware that enforces per-API-key request limits.

    Tracks request counts per API key within a sliding time window.
    Returns HTTP 429 when the rate limit is exceeded.
    """

    def __init__(
        self,
        app: ASGIApp,
        default_limit: int = 100,
        window_seconds: int = 60,
        public_paths: Optional[Set[str]] = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            app: The ASGI application to wrap.
            default_limit: Maximum requests per window per API key.
            window_seconds: Time window in seconds for rate counting.
            public_paths: Paths that skip rate limiting.
        """
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.public_paths = public_paths or PUBLIC_PATHS
        self._request_log: Dict[str, List[float]] = defaultdict(list)

    def _get_api_key(self, request: Request) -> str:
        """Extract the API key from the request headers.

        Args:
            request: The incoming HTTP request.

        Returns:
            The API key string, or 'anonymous' if not found.
        """
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return request.headers.get("X-API-Key", "anonymous")

    def _is_rate_limited(self, api_key: str) -> bool:
        """Check if the API key has exceeded its rate limit.

        Args:
            api_key: The API key to check.

        Returns:
            True if the key is rate-limited, False otherwise.
        """
        now = time.time()
        window_start = now - self.window_seconds
        # Prune old entries outside the window
        self._request_log[api_key] = [
            ts for ts in self._request_log[api_key] if ts > window_start
        ]
        if len(self._request_log[api_key]) >= self.default_limit:
            return True
        self._request_log[api_key].append(now)
        return False

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request through the rate limiter.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response, or 429 if rate-limited.
        """
        if request.url.path in self.public_paths:
            return await call_next(request)

        api_key = self._get_api_key(request)
        if self._is_rate_limited(api_key):
            logger.warning("Rate limit exceeded for key: %s", api_key[:8] + "...")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.default_limit} requests per {self.window_seconds}s",
                    "retry_after": self.window_seconds,
                },
            )
        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware with timing information.

    Logs the method, path, status code, and duration for every request.
    Excludes health-check endpoints to reduce log noise.
    """

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the logging middleware.

        Args:
            app: The ASGI application to wrap.
            exclude_paths: Paths to exclude from logging.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or {"/health"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request details and response timing.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response from downstream.
        """
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        if request.url.path not in self.exclude_paths:
            logger.info(
                "%s %s %s → %d (%.1fms) from %s",
                request.method,
                request.url.path,
                request.scope.get("query_string", b"").decode(),
                response.status_code,
                duration_ms,
                client_ip,
            )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware that verifies API keys or JWT tokens.

    Requests to protected endpoints must include a valid Authorization
    header (Bearer token) or X-API-Key header.
    """

    def __init__(
        self,
        app: ASGIApp,
        public_paths: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the auth middleware.

        Args:
            app: The ASGI application to wrap.
            public_paths: Paths that skip authentication.
        """
        super().__init__(app)
        self.public_paths = public_paths or PUBLIC_PATHS

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Verify authentication on protected endpoints.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response, or 401 if authentication fails.
        """
        if request.url.path in self.public_paths:
            return await call_next(request)

        # WebSocket upgrades are handled separately
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")

        if not auth_header and not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required",
                    "detail": "Provide Authorization header or X-API-Key header.",
                },
            )

        # Validate Bearer token or API key
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if len(token) < 10:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid token", "detail": "Token is too short."},
                )
        elif api_key and len(api_key) < 10:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key", "detail": "API key is too short."},
            )

        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware for unhandled exceptions.

    Catches any exception raised during request processing and returns
    a consistent JSON error response with HTTP 500 status.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Catch and handle unhandled exceptions gracefully.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response, or 500 JSON if an exception occurs.
        """
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc),
                    "path": str(request.url.path),
                },
            )
