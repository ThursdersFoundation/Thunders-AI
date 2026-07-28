"""OpenAI API client for Thunders AI.

Provides a high-level client for OpenAI-compatible chat, completion,
and embedding endpoints with streaming, rate limiting, and retry logic.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)

# Default models supported by this client
SUPPORTED_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
]


class RateLimiter:
    """Simple token-bucket rate limiter.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute.
        tokens_per_minute: Maximum tokens allowed per minute.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 150_000,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._request_timestamps: List[float] = []
        self._token_usage: List[Tuple[float, int]] = []

    def allow_request(self, estimated_tokens: int = 0) -> bool:
        """Check whether a request is permitted under current limits."""
        now = time.time()
        window = 60.0

        # Prune old entries
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < window
        ]
        self._token_usage = [
            (t, tok) for t, tok in self._token_usage if now - t < window
        ]

        if len(self._request_timestamps) >= self.requests_per_minute:
            return False
        total_tokens = sum(tok for _, tok in self._token_usage)
        if total_tokens + estimated_tokens > self.tokens_per_minute:
            return False

        return True

    def record_request(self, tokens_used: int = 0) -> None:
        """Record a completed request for rate tracking."""
        now = time.time()
        self._request_timestamps.append(now)
        if tokens_used > 0:
            self._token_usage.append((now, tokens_used))


class OpenAIClient:
    """Client for OpenAI-compatible APIs.

    Supports GPT-4, GPT-3.5, and custom models with streaming,
    rate limiting, and automatic retry on transient failures.

    Attributes:
        model: Default model identifier.
        api_key: API key for authentication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        rpm: int = 60,
        tpm: int = 150_000,
    ) -> None:
        if model not in SUPPORTED_MODELS and not model.startswith("ft:"):
            logger.warning("Model '%s' is not in the known list; proceeding anyway", model)

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._rate_limiter = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm)
        self._session_id = f"oai-{uuid.uuid4().hex[:8]}"

        logger.info(
            "OpenAIClient initialised: model=%s, base_url=%s", model, base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Override the default model.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in the response.
            stream: If True, return a generator of chunk dicts.
            **kwargs: Additional API parameters.

        Returns:
            Response dict, or generator of chunk dicts if streaming.

        Raises:
            ValueError: If messages list is empty.
            RuntimeError: If all retry attempts fail.
        """
        if not messages:
            raise ValueError("messages must be a non-empty list")

        model = model or self.model
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages) + max_tokens

        if not self._rate_limiter.allow_request(estimated_tokens):
            raise RuntimeError("Rate limit exceeded; please retry later")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._stream_chat(payload)

        response = self._request_with_retry("POST", "/chat/completions", payload)
        self._rate_limiter.record_request(
            response.get("usage", {}).get("total_tokens", estimated_tokens)
        )
        return response

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a legacy text completion request.

        Args:
            prompt: The text prompt.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional API parameters.

        Returns:
            Completion response dict.

        Raises:
            ValueError: If prompt is empty.
        """
        if not prompt:
            raise ValueError("prompt must be a non-empty string")

        model = model or self.model
        estimated_tokens = len(prompt) // 4 + max_tokens

        if not self._rate_limiter.allow_request(estimated_tokens):
            raise RuntimeError("Rate limit exceeded; please retry later")

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = self._request_with_retry("POST", "/completions", payload)
        self._rate_limiter.record_request(
            response.get("usage", {}).get("total_tokens", estimated_tokens)
        )
        return response

    def embed(
        self,
        input_texts: List[str],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate embeddings for a list of texts.

        Args:
            input_texts: Texts to embed.
            model: Embedding model name.
            dimensions: Optional output dimensionality.

        Returns:
            Embedding response with vectors and usage.

        Raises:
            ValueError: If input_texts is empty.
        """
        if not input_texts:
            raise ValueError("input_texts must be a non-empty list")

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_texts,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = self._request_with_retry("POST", "/embeddings", payload)
        self._rate_limiter.record_request(
            response.get("usage", {}).get("total_tokens", len(input_texts) * 50)
        )
        return response

    # -- Internal helpers ---------------------------------------------------

    def _stream_chat(
        self, payload: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """Simulate streaming chat responses (placeholder for real SSE)."""
        yield {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": "[streaming placeholder]"}, "index": 0}],
        }

    def _request_with_retry(
        self, method: str, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an HTTP request with exponential-backoff retry.

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            payload: Request body.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: After all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._make_request(method, endpoint, payload)
            except Exception as exc:
                last_error = exc
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"All {self.max_retries} retry attempts failed: {last_error}"
        )

    def _make_request(
        self, method: str, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate an HTTP request to the OpenAI API.

        In production, this would use ``httpx`` or ``aiohttp``.
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug("Request: %s %s", method, url)

        # Simulated response
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": payload.get("model", self.model),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Simulated response for: {payload.get('messages', [{}])[0].get('content', '')[:50]}...]",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50,
            },
        }
