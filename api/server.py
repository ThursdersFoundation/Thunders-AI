"""Thunders AI API Server - FastAPI application.

Provides the main FastAPI application instance with CORS middleware,
lifecycle event handlers, health check endpoint, and uvicorn configuration.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware import (
    RateLimitMiddleware,
    LoggingMiddleware,
    AuthMiddleware,
    ErrorHandlingMiddleware,
)
from api.routes import router
from api.websocket import ConnectionManager

logger = logging.getLogger(__name__)

# Global connection manager for WebSocket clients
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown events.

    On startup, initializes the connection manager and logs server readiness.
    On shutdown, gracefully disconnects all WebSocket clients and cleans up.
    """
    logger.info("Thunders AI API Server starting up...")
    await connection_manager.initialize()
    logger.info("Connection manager initialized successfully")
    yield
    logger.info("Thunders AI API Server shutting down...")
    await connection_manager.shutdown()
    logger.info("All connections closed. Server stopped.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        FastAPI: Fully configured application with middleware, routes,
            and lifecycle handlers.
    """
    application = FastAPI(
        title="Thunders AI API",
        description=(
            "Production-grade AI API supporting chat completions, vision analysis, "
            "speech processing, and robotics navigation. Compatible with OpenAI API format."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS Middleware ---
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Custom Middleware Stack ---
    application.add_middleware(ErrorHandlingMiddleware)
    application.add_middleware(AuthMiddleware)
    application.add_middleware(LoggingMiddleware)
    application.add_middleware(RateLimitMiddleware)

    # --- Include API Routes ---
    application.include_router(router, prefix="/api/v1")

    # --- Health Check Endpoint ---
    @application.get("/health", tags=["Health"])
    async def health_check() -> JSONResponse:
        """Return the health status of the API server.

        Returns:
            JSONResponse: Server health information including status,
                version, and uptime.
        """
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "version": "1.0.0",
                "service": "thunders-ai-api",
                "timestamp": time.time(),
            },
        )

    # --- Root Endpoint ---
    @application.get("/", tags=["Root"])
    async def root() -> JSONResponse:
        """Root endpoint providing API overview and documentation links.

        Returns:
            JSONResponse: Welcome message with links to docs and health check.
        """
        return JSONResponse(
            status_code=200,
            content={
                "message": "Welcome to Thunders AI API",
                "documentation": "/docs",
                "health_check": "/health",
                "api_version": "v1",
            },
        )

    return application


# Create the default application instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info",
    reload: bool = False,
) -> None:
    """Run the API server using uvicorn.

    Args:
        host: Bind address for the server.
        port: Port number for the server.
        workers: Number of worker processes.
        log_level: Logging level (debug, info, warning, error, critical).
        reload: Enable auto-reload for development.
    """
    logger.info("Starting Thunders AI API Server on %s:%d", host, port)
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
    )


if __name__ == "__main__":
    run_server(reload=True)
