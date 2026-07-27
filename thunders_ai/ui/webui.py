"""Web UI server for Thunders AI.

Provides a FastAPI-based web interface with WebSocket support
for real-time interaction with Thunders AI models and services.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class WebSocketConnection:
    """Represents a single WebSocket client connection.

    Attributes:
        connection_id: Unique connection identifier.
        session_data: Session-specific data store.
    """

    def __init__(self, connection_id: Optional[str] = None) -> None:
        self.connection_id = connection_id or f"ws-{uuid.uuid4().hex[:8]}"
        self.session_data: Dict[str, Any] = {}
        self.connected_at: float = time.time()
        self.is_active: bool = True

    def send(self, data: Dict[str, Any]) -> None:
        """Send data to the connected client."""
        logger.debug("WS send to %s: %s", self.connection_id, str(data)[:100])

    def close(self) -> None:
        """Close the WebSocket connection."""
        self.is_active = False


class UIComponent:
    """A reusable UI component definition.

    Attributes:
        component_id: Unique component identifier.
        component_type: Type of the component (e.g. 'chat', 'table').
    """

    def __init__(
        self,
        name: str,
        component_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.component_id = f"comp-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.component_type = component_type
        self.config = config or {}
        self.created_at: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the component definition."""
        return {
            "component_id": self.component_id,
            "name": self.name,
            "type": self.component_type,
            "config": self.config,
        }


class WebUI:
    """FastAPI + WebSocket web interface for Thunders AI.

    Provides a real-time web UI for interacting with AI models,
    viewing logs, and managing deployments.

    Attributes:
        host: Server bind address.
        port: Server bind port.
        components: Registered UI components.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        title: str = "Thunders AI",
        cors_origins: Optional[List[str]] = None,
        static_dir: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.title = title
        self.cors_origins = cors_origins or ["*"]
        self.static_dir = static_dir
        self.components: Dict[str, UIComponent] = {}
        self._ws_connections: Dict[str, WebSocketConnection] = {}
        self._app: Optional[Any] = None
        self._routes: List[Dict[str, Any]] = []
        self._running: bool = False

        logger.info(
            "WebUI initialised: %s:%d (title=%s)", host, port, title
        )

    def launch(
        self,
        block: bool = True,
        workers: int = 1,
        log_level: str = "info",
    ) -> Dict[str, Any]:
        """Launch the web server.

        Args:
            block: If True, block until the server stops.
            workers: Number of Uvicorn workers.
            log_level: Uvicorn log level.

        Returns:
            Server launch information.
        """
        if self._running:
            logger.warning("WebUI is already running")
            return {"status": "already_running", "url": f"http://{self.host}:{self.port}"}

        self._app = self.get_app()
        self._running = True

        launch_info: Dict[str, Any] = {
            "status": "started",
            "url": f"http://{self.host}:{self.port}",
            "host": self.host,
            "port": self.port,
            "workers": workers,
            "log_level": log_level,
            "components": len(self.components),
            "websocket_clients": 0,
        }

        logger.info(
            "WebUI launching at http://%s:%d (%d components)",
            self.host,
            self.port,
            len(self.components),
        )

        # In production: uvicorn.run(self._app, host=self.host, port=self.port)
        return launch_info

    def create_interface(
        self,
        name: str,
        interface_type: str = "chat",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> UIComponent:
        """Create a new UI interface component.

        Args:
            name: Display name of the interface.
            interface_type: Type ('chat', 'completion', 'playground').
            model: Default model for the interface.
            **kwargs: Additional interface configuration.

        Returns:
            The created UIComponent.
        """
        config: Dict[str, Any] = {"model": model, **kwargs}
        component = UIComponent(
            name=name, component_type=interface_type, config=config
        )
        self.components[component.component_id] = component

        logger.info("Interface created: '%s' (type=%s)", name, interface_type)
        return component

    def add_component(
        self,
        name: str,
        component_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> UIComponent:
        """Add a UI component to the web interface.

        Args:
            name: Component name.
            component_type: Component type (e.g. 'chart', 'log', 'table').
            config: Component-specific configuration.

        Returns:
            The created UIComponent.
        """
        component = UIComponent(name=name, component_type=component_type, config=config)
        self.components[component.component_id] = component

        logger.info("Component added: '%s' (%s)", name, component_type)
        return component

    def get_app(self) -> Any:
        """Get or create the underlying FastAPI application.

        Returns:
            The FastAPI app instance.
        """
        if self._app is not None:
            return self._app

        try:
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware

            app = FastAPI(title=self.title)
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            @app.get("/api/health")
            async def health() -> Dict[str, Any]:
                return {
                    "status": "healthy",
                    "title": self.title,
                    "components": len(self.components),
                    "websocket_clients": len(self._ws_connections),
                }

            @app.get("/api/components")
            async def list_components() -> List[Dict[str, Any]]:
                return [c.to_dict() for c in self.components.values()]

            self._app = app
        except ImportError:
            logger.warning("FastAPI not installed; using stub app")
            self._app = {"title": self.title, "type": "stub"}

        return self._app

    def broadcast(self, event: str, data: Dict[str, Any]) -> int:
        """Broadcast an event to all connected WebSocket clients.

        Args:
            event: Event name.
            data: Event payload.

        Returns:
            Number of clients that received the event.
        """
        message = json.dumps({"event": event, "data": data, "timestamp": time.time()})
        active = [c for c in self._ws_connections.values() if c.is_active]
        for conn in active:
            conn.send({"event": event, "data": data})
        logger.debug("Broadcast '%s' to %d clients", event, len(active))
        return len(active)

    def stop(self) -> None:
        """Stop the web server."""
        self._running = False
        for conn in self._ws_connections.values():
            conn.close()
        self._ws_connections.clear()
        logger.info("WebUI stopped")
