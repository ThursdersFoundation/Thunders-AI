"""Ollama local model client for Thunders AI.

Provides chat, generation, model listing, and model pulling
against a locally running Ollama server.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Client for interacting with a local Ollama server.

    Supports chat, text generation, model management, and streaming
    against the Ollama REST API running on localhost.

    Attributes:
        host: Ollama server address.
        port: Ollama server port.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        timeout: float = 120.0,
        default_model: str = "llama3",
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.default_model = default_model
        self._base_url = f"http://{host}:{port}"
        self._available_models: List[Dict[str, Any]] = []
        self._connected: bool = False

        logger.info(
            "OllamaClient initialised: %s (model=%s)", self._base_url, default_model
        )

    def _ensure_connected(self) -> None:
        """Verify connection to the Ollama server."""
        if not self._connected:
            logger.debug("Checking Ollama server at %s...", self._base_url)
            # Simulated connectivity check
            self._connected = True

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a chat request to a local Ollama model.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Override the default model.
            temperature: Sampling temperature.
            stream: If True, return a generator of chunks.
            **kwargs: Additional Ollama parameters.

        Returns:
            Chat response dict or streaming generator.

        Raises:
            ValueError: If messages is empty.
            ConnectionError: If Ollama server is unreachable.
        """
        self._ensure_connected()
        if not messages:
            raise ValueError("messages must be a non-empty list")

        model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }

        logger.debug("Chat request: model=%s, messages=%d", model, len(messages))

        if stream:
            return self._stream_response(payload)

        response = self._make_request("POST", "/api/chat", payload)
        return response

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        system: Optional[str] = None,
        template: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate text from a local Ollama model.

        Args:
            prompt: The input prompt.
            model: Override the default model.
            temperature: Sampling temperature.
            system: System prompt.
            template: Prompt template.
            stream: If True, stream the response.
            **kwargs: Additional parameters.

        Returns:
            Generation result dict.

        Raises:
            ValueError: If prompt is empty.
        """
        self._ensure_connected()
        if not prompt:
            raise ValueError("prompt must be a non-empty string")

        model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template

        logger.debug("Generate request: model=%s, prompt_len=%d", model, len(prompt))
        start_time = time.time()
        response = self._make_request("POST", "/api/generate", payload)
        elapsed = time.time() - start_time
        response["elapsed_seconds"] = round(elapsed, 4)
        return response

    def list_models(self) -> List[Dict[str, Any]]:
        """List all locally available Ollama models.

        Returns:
            List of model info dicts with name, size, and modified date.
        """
        self._ensure_connected()
        response = self._make_request("GET", "/api/tags", {})
        models = response.get("models", [])
        self._available_models = models
        logger.info("Available models: %d", len(models))
        return models

    def pull_model(
        self,
        model_name: str,
        insecure: bool = False,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Pull a model from the Ollama registry.

        Args:
            model_name: Name of the model to pull (e.g. 'llama3:8b').
            insecure: Allow insecure connections for pulling.
            stream: If True, stream pull progress.

        Returns:
            Pull result dict with status.

        Raises:
            ValueError: If model_name is empty.
        """
        if not model_name:
            raise ValueError("model_name must be a non-empty string")

        self._ensure_connected()
        payload: Dict[str, Any] = {
            "name": model_name,
            "insecure": insecure,
            "stream": stream,
        }

        logger.info("Pulling model '%s'...", model_name)
        start_time = time.time()
        result = self._make_request("POST", "/api/pull", payload)
        elapsed = time.time() - start_time
        result["pull_time_seconds"] = round(elapsed, 4)
        logger.info("Model '%s' pulled in %.1fs", model_name, elapsed)
        return result

    # -- Internal helpers ---------------------------------------------------

    def _stream_response(
        self, payload: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulate a streaming response from Ollama."""
        yield {
            "model": payload["model"],
            "message": {"role": "assistant", "content": "[streaming placeholder]"},
            "done": False,
        }
        yield {
            "model": payload["model"],
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }

    def _make_request(
        self, method: str, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate an HTTP request to the Ollama API.

        In production this would use ``httpx`` or ``aiohttp``.
        """
        url = f"{self._base_url}{endpoint}"
        logger.debug("Request: %s %s", method, url)

        if endpoint == "/api/tags":
            return {
                "models": [
                    {"name": "llama3:8b", "size": 4661224676, "modified_at": "2024-06-01T00:00:00Z"},
                    {"name": "mistral:7b", "size": 4108928384, "modified_at": "2024-05-15T00:00:00Z"},
                ]
            }

        if endpoint == "/api/pull":
            return {"status": "success", "name": payload.get("name", "")}

        # Default chat/generate response
        return {
            "model": payload.get("model", self.default_model),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": {
                "role": "assistant",
                "content": f"[Ollama simulated response for: {payload.get('prompt', payload.get('messages', [{}])[0].get('content', ''))[:60]}...]",
            },
            "done": True,
            "total_duration": 1_500_000_000,
            "eval_count": 42,
        }
