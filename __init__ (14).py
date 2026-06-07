"""Thunders AI API Module.

Provides a FastAPI-based REST API and WebSocket server for interacting
with Thunders AI models, vision systems, speech processing, and robotics.
"""

from api.server import create_app, app
from api.authentication import APIKeyAuth, JWTAuth, create_api_key, verify_api_key
from api.routes import router
from api.websocket import ConnectionManager

__all__ = [
    "create_app",
    "app",
    "APIKeyAuth",
    "JWTAuth",
    "create_api_key",
    "verify_api_key",
    "router",
    "ConnectionManager",
]

__version__ = "1.0.0"
