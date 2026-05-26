"""Chat model with streaming, sampling controls, and conversation management."""

from __future__ import annotations

import time
from typing import Any, Dict, Generator, List, Optional

from thunders_ai.config import ThundersConfig
from thunders_ai.core.engine import Engine
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class ChatMessage:
    """Structured representation of a single chat message."""

    __slots__ = ("role", "content", "timestamp", "metadata")

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, str]:
        """Return a plain dict with role and content."""
        return {"role": self.role, "content": self.content}


class ChatModel:
    """High-level chat interface backed by a Thunders AI Engine.

    Supports system prompts, streaming, sampling parameters, and
    automatic conversation history management.

    Args:
        config: ThundersConfig instance.
        engine: A :class:`~thunders_ai.core.engine.Engine` used for generation.

    Example::

        chat = ChatModel(config, engine)
        reply = chat.chat("What is quantum computing?")
        for chunk in chat.stream("Explain further"):
            print(chunk, end="")
    """

    def __init__(self, config: ThundersConfig, engine: Engine) -> None:
        self._config = config
        self._engine = engine
        self._system_prompt: Optional[str] = getattr(config, "system_prompt", None)
        self._history: List[ChatMessage] = []
        self._max_history: int = getattr(config, "max_chat_history", 50)
        self._default_temperature: float = getattr(config, "temperature", 0.7)
        self._default_top_p: float = getattr(config, "top_p", 0.9)
        self._default_top_k: int = getattr(config, "top_k", 50)
        self._default_max_tokens: int = getattr(config, "max_new_tokens", 512)
        logger.info("ChatModel initialized.")

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        """Set or replace the system prompt.

        Args:
            prompt: New system prompt text.
        """
        self._system_prompt = prompt
        logger.debug("System prompt updated (%d chars).", len(prompt))

    # ------------------------------------------------------------------
    # Chat (single-turn)
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Send a user message and return the assistant's reply.

        Conversation history is maintained automatically.

        Args:
            message: User message text.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling threshold override.
            top_k: Top-k sampling override.
            max_new_tokens: Max generation length override.
            **kwargs: Additional generation kwargs.

        Returns:
            Assistant reply string.
        """
        self._add_message("user", message)
        prompt = self._build_prompt()

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self._default_max_tokens,
            "temperature": temperature or self._default_temperature,
            "top_p": top_p or self._default_top_p,
            "top_k": top_k or self._default_top_k,
            **kwargs,
        }

        reply = self._engine.generate(prompt, **gen_kwargs)
        self._add_message("assistant", reply)
        return reply

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        message: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        chunk_size: int = 1,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Send a user message and yield the response as a stream of chunks.

        Args:
            message: User message text.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling threshold override.
            top_k: Top-k sampling override.
            max_new_tokens: Max generation length override.
            chunk_size: Number of tokens per yielded chunk.
            **kwargs: Additional generation kwargs.

        Yields:
            Text chunks of the assistant's reply.
        """
        self._add_message("user", message)
        prompt = self._build_prompt()

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self._default_max_tokens,
            "temperature": temperature or self._default_temperature,
            "top_p": top_p or self._default_top_p,
            "top_k": top_k or self._default_top_k,
            **kwargs,
        }

        full_response: List[str] = []
        for chunk in self._engine.generate_stream(prompt, chunk_size=chunk_size, **gen_kwargs):
            full_response.append(chunk)
            yield chunk

        self._add_message("assistant", "".join(full_response))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _add_message(self, role: str, content: str) -> None:
        """Append a message and enforce the history limit."""
        self._history.append(ChatMessage(role=role, content=content))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _build_prompt(self) -> str:
        """Construct the full prompt including system prompt and history."""
        parts: List[str] = []
        if self._system_prompt:
            parts.append(f"[System] {self._system_prompt}")
        for msg in self._history:
            tag = msg.role.capitalize()
            parts.append(f"[{tag}] {msg.content}")
        parts.append("[Assistant]")
        return "\n".join(parts)

    def get_history(self) -> List[Dict[str, str]]:
        """Return conversation history as a list of dicts."""
        return [m.to_dict() for m in self._history]

    def clear_history(self) -> None:
        """Erase the conversation history."""
        self._history.clear()
        logger.info("Chat history cleared.")

    @property
    def message_count(self) -> int:
        """Number of messages in the current history."""
        return len(self._history)
