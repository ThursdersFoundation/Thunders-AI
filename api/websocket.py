"""Thunders AI WebSocket Handler - Real-time communication support.

Provides WebSocket endpoints for real-time chat and streaming responses,
a connection manager for handling multiple concurrent clients, and
heartbeat/ping-pong keepalive support.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# Heartbeat interval in seconds
HEARTBEAT_INTERVAL: float = 30.0
# Maximum time without heartbeat response before disconnect (seconds)
HEARTBEAT_TIMEOUT: float = 60.0


@dataclass
class ClientConnection:
    """Represents a single WebSocket client connection.

    Attributes:
        websocket: The WebSocket connection instance.
        client_id: Unique identifier for this client.
        connected_at: Timestamp when the connection was established.
        last_heartbeat: Timestamp of the last received heartbeat.
        metadata: Optional metadata associated with the client.
    """
    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionManager:
    """Manages multiple WebSocket client connections.

    Provides methods for connecting, disconnecting, broadcasting,
    and sending targeted messages to individual clients. Includes
    heartbeat monitoring to detect and remove stale connections.
    """

    def __init__(self) -> None:
        """Initialize the connection manager with empty client list."""
        self._clients: Dict[str, ClientConnection] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Start the heartbeat monitoring background task."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        logger.info("Connection manager heartbeat monitor started")

    async def shutdown(self) -> None:
        """Gracefully disconnect all clients and cancel heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        for client_id, client in list(self._clients.items()):
            try:
                await client.websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Server shutting down")
            except Exception:
                pass
        self._clients.clear()
        logger.info("Connection manager shut down, all clients disconnected")

    async def connect(self, websocket: WebSocket, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The incoming WebSocket connection.
            metadata: Optional metadata to associate with the client.

        Returns:
            The unique client ID assigned to the connection.
        """
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self._clients[client_id] = ClientConnection(
            websocket=websocket,
            client_id=client_id,
            metadata=metadata or {},
        )
        logger.info("Client %s connected. Total clients: %d", client_id[:8], len(self._clients))
        await self.send_to_client(client_id, {
            "type": "connected",
            "client_id": client_id,
            "message": "Successfully connected to Thunders AI WebSocket server.",
        })
        return client_id

    def disconnect(self, client_id: str) -> None:
        """Remove a client from the connection manager.

        Args:
            client_id: The ID of the client to disconnect.
        """
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info("Client %s disconnected. Total clients: %d", client_id[:8], len(self._clients))

    async def send_to_client(self, client_id: str, data: Dict[str, Any]) -> None:
        """Send a JSON message to a specific client.

        Args:
            client_id: The target client ID.
            data: Dictionary payload to serialize and send.
        """
        client = self._clients.get(client_id)
        if client:
            try:
                await client.websocket.send_json(data)
            except Exception as exc:
                logger.error("Failed to send to client %s: %s", client_id[:8], exc)
                self.disconnect(client_id)

    async def broadcast(self, data: Dict[str, Any], exclude: Optional[List[str]] = None) -> None:
        """Broadcast a JSON message to all connected clients.

        Args:
            data: Dictionary payload to broadcast.
            exclude: Optional list of client IDs to exclude.
        """
        exclude_set = set(exclude or [])
        for client_id in list(self._clients.keys()):
            if client_id not in exclude_set:
                await self.send_to_client(client_id, data)

    @property
    def active_connections(self) -> int:
        """Return the number of currently connected clients."""
        return len(self._clients)

    async def _heartbeat_monitor(self) -> None:
        """Background task that sends heartbeat pings and checks timeouts."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                now = time.time()
                stale_clients = [
                    cid for cid, client in self._clients.items()
                    if now - client.last_heartbeat > HEARTBEAT_TIMEOUT
                ]
                for client_id in stale_clients:
                    logger.warning("Client %s timed out, disconnecting", client_id[:8])
                    client = self._clients.get(client_id)
                    if client:
                        try:
                            await client.websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Heartbeat timeout")
                        except Exception:
                            pass
                    self.disconnect(client_id)

                # Send heartbeat ping to all active clients
                await self.broadcast({"type": "ping", "timestamp": now})
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat monitor error: %s", exc)


# Global connection manager instance
manager = ConnectionManager()


@ws_router.websocket("/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time chat communication.

    Accepts messages in JSON format with a 'message' field and returns
    AI-generated responses. Supports heartbeat pong responses.

    Message format (incoming):
        {"type": "message", "content": "Hello!"}
        {"type": "pong"}

    Message format (outgoing):
        {"type": "response", "content": "...", "model": "thunders-7b"}
    """
    client_id = await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to_client(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format.",
                })
                continue

            msg_type = data.get("type", "")

            # Handle heartbeat pong
            if msg_type == "pong":
                client = manager._clients.get(client_id)
                if client:
                    client.last_heartbeat = time.time()
                continue

            # Handle chat messages
            if msg_type == "message":
                content = data.get("content", "")
                logger.info("Chat WS from %s: %s", client_id[:8], content[:80])
                # Placeholder response — integrate with ThundersAI model here
                await manager.send_to_client(client_id, {
                    "type": "response",
                    "content": f"Echo: {content}",
                    "model": "thunders-7b",
                    "timestamp": time.time(),
                })
            else:
                await manager.send_to_client(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info("Chat WebSocket client %s disconnected", client_id[:8])


@ws_router.websocket("/stream")
async def stream_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming AI response tokens.

    Accepts a prompt and streams back individual tokens as they
    are generated, enabling real-time display of AI output.

    Message format (incoming):
        {"type": "prompt", "content": "Explain quantum computing", "model": "thunders-7b"}

    Message format (outgoing, per token):
        {"type": "token", "content": "word", "index": 0}
    Message format (outgoing, completion):
        {"type": "done", "total_tokens": 42}
    """
    client_id = await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to_client(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format.",
                })
                continue

            msg_type = data.get("type", "")

            # Handle heartbeat pong
            if msg_type == "pong":
                client = manager._clients.get(client_id)
                if client:
                    client.last_heartbeat = time.time()
                continue

            # Handle streaming prompt
            if msg_type == "prompt":
                content = data.get("content", "")
                model = data.get("model", "thunders-7b")
                logger.info("Stream WS from %s: %s", client_id[:8], content[:80])

                # Placeholder streaming — integrate with ThundersAI model here
                tokens = content.split()
                for idx, token in enumerate(tokens):
                    await manager.send_to_client(client_id, {
                        "type": "token",
                        "content": token,
                        "index": idx,
                    })
                    await asyncio.sleep(0.05)  # Simulate generation delay

                await manager.send_to_client(client_id, {
                    "type": "done",
                    "total_tokens": len(tokens),
                    "model": model,
                })
            else:
                await manager.send_to_client(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info("Stream WebSocket client %s disconnected", client_id[:8])
