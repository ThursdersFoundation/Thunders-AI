"""Inference engine with batch processing, caching, and streaming support."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class OutputFormat(str, Enum):
    """Supported inference output formats."""
    TEXT = "text"
    JSON = "json"
    TOKENS = "tokens"
    LOGITS = "logits"


class InferenceEngine:
    """High-performance inference engine with caching and streaming.

    Wraps a model/endpoint to provide batch inference, result caching,
    streaming generation, and multiple output formats.

    Args:
        config: ThundersConfig instance.
        engine: A :class:`~thunders_ai.core.engine.Engine` instance used for
            generation when no custom callable is supplied.

    Example::

        inf = InferenceEngine(config, engine)
        result = inf.run("Translate to French: Hello")
        for token in inf.stream("Tell me a story"):
            print(token, end="")
    """

    _MAX_CACHE_SIZE = 1024

    def __init__(self, config: ThundersConfig, engine: Any) -> None:
        self._config = config
        self._engine = engine
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._default_max_tokens: int = getattr(config, "max_new_tokens", 256)
        self._default_temperature: float = getattr(config, "temperature", 0.7)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(prompt: str, **kwargs: Any) -> str:
        """Produce a deterministic cache key from prompt and generation params."""
        raw = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Return cached result or *None*; moves hit to MRU position."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _put_cached(self, key: str, result: Dict[str, Any]) -> None:
        """Store a result in cache, evicting LRU entries when full."""
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self._MAX_CACHE_SIZE:
            evicted = self._cache.popitem(last=False)
            logger.debug("Cache eviction: %s", evicted[0][:16])

    def clear_cache(self) -> None:
        """Clear the entire inference cache."""
        self._cache.clear()
        logger.info("Inference cache cleared.")

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        output_format: OutputFormat = OutputFormat.TEXT,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], List[int]]:
        """Run inference on a single prompt.

        Args:
            prompt: The input text.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            output_format: Desired output format.
            use_cache: Whether to check / store cache.
            **kwargs: Extra generation kwargs.

        Returns:
            Generated output in the requested format.
        """
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self._default_max_tokens,
            "temperature": temperature or self._default_temperature,
            **kwargs,
        }
        cache_key = self._cache_key(prompt, **gen_kwargs)

        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug("Cache hit for prompt '%s…'", prompt[:32])
                return self._format_output(cached["text"], cached.get("tokens", []), output_format)

        t0 = time.perf_counter()
        text = self._engine.generate(prompt, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("Inference completed in %.2fs", elapsed)

        tokens = list(range(len(text.split())))  # placeholder token ids
        result = {"text": text, "tokens": tokens, "elapsed": elapsed}

        if use_cache:
            self._put_cached(cache_key, result)

        return self._format_output(text, tokens, output_format)

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        output_format: OutputFormat = OutputFormat.TEXT,
        **kwargs: Any,
    ) -> List[Union[str, Dict[str, Any], List[int]]]:
        """Run inference on a list of prompts sequentially.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            output_format: Desired output format.
            **kwargs: Extra generation kwargs.

        Returns:
            List of outputs, one per prompt.
        """
        results: List[Any] = []
        for prompt in prompts:
            results.append(
                self.run(prompt, max_new_tokens, temperature, output_format, **kwargs)
            )
        logger.info("Batch inference: %d prompts processed.", len(prompts))
        return results

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chunk_size: int = 1,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Yield generated tokens incrementally (simulated streaming).

        Args:
            prompt: Input text.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            chunk_size: Number of tokens per yielded chunk.
            **kwargs: Extra generation kwargs.

        Yields:
            Chunks of generated text.
        """
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self._default_max_tokens,
            "temperature": temperature or self._default_temperature,
            **kwargs,
        }
        full_text = self._engine.generate(prompt, **gen_kwargs)
        words = full_text.split(" ")
        buffer: List[str] = []
        for word in words:
            buffer.append(word)
            if len(buffer) >= chunk_size:
                yield " ".join(buffer) + " "
                buffer.clear()
        if buffer:
            yield " ".join(buffer)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        text: str,
        tokens: List[int],
        fmt: OutputFormat,
    ) -> Union[str, Dict[str, Any], List[int]]:
        """Convert raw output to the requested format."""
        if fmt == OutputFormat.TEXT:
            return text
        if fmt == OutputFormat.JSON:
            return {"text": text, "token_count": len(tokens)}
        if fmt == OutputFormat.TOKENS:
            return tokens
        if fmt == OutputFormat.LOGITS:
            return {"text": text, "logits": None}
        return text
