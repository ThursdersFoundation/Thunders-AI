"""General-purpose helper utilities for Thunders AI.

Provides decorators and utility functions for timing, retries,
list processing, dictionary manipulation, ID generation, and
byte formatting.
"""

from __future__ import annotations

import functools
import hashlib
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from thunders_ai.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class Helpers:
    """Collection of static helper methods and decorators.

    All methods are static and stateless, designed for reuse
    across the Thunders AI codebase.
    """

    @staticmethod
    def timing(func: Callable[..., T]) -> Callable[..., T]:
        """Decorator that measures and logs function execution time.

        Args:
            func: The function to time.

        Returns:
            Wrapped function that logs elapsed time.

        Example::

            @Helpers.timing
            def slow_operation():
                time.sleep(1)
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(
                "%s executed in %.4f seconds", func.__qualname__, elapsed
            )
            return result

        return wrapper

    @staticmethod
    def retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[type, ...] = (Exception,),
    ) -> Callable[..., Callable[..., T]]:
        """Decorator that retries a function on specified exceptions.

        Args:
            max_attempts: Maximum number of attempts.
            delay: Initial delay between retries in seconds.
            backoff: Multiplier applied to delay after each retry.
            exceptions: Tuple of exception types to catch.

        Returns:
            Decorator function.

        Example::

            @Helpers.retry(max_attempts=5, delay=0.5)
            def flaky_api_call():
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                current_delay = delay
                last_exception: Optional[Exception] = None

                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:
                        last_exception = exc
                        if attempt < max_attempts:
                            logger.warning(
                                "%s failed (attempt %d/%d): %s; "
                                "retrying in %.1fs",
                                func.__qualname__,
                                attempt,
                                max_attempts,
                                exc,
                                current_delay,
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(
                                "%s failed after %d attempts: %s",
                                func.__qualname__,
                                max_attempts,
                                exc,
                            )

                raise last_exception  # type: ignore[misc]

            return wrapper

        return decorator

    @staticmethod
    def chunk_list(
        items: List[T],
        chunk_size: int,
    ) -> List[List[T]]:
        """Split a list into fixed-size chunks.

        Args:
            items: The list to chunk.
            chunk_size: Maximum size of each chunk.

        Returns:
            List of chunked sub-lists.

        Raises:
            ValueError: If chunk_size is less than 1.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        return [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

    @staticmethod
    def flatten_dict(
        d: Dict[str, Any],
        parent_key: str = "",
        separator: str = ".",
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary into dot-separated keys.

        Args:
            d: The dictionary to flatten.
            parent_key: Prefix for keys (used in recursion).
            separator: Key separator string.

        Returns:
            Flattened dictionary with compound keys.

        Example::

            >>> Helpers.flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
            {'a.b': 1, 'a.c.d': 2}
        """
        items: List[Tuple[str, Any]] = []
        for key, value in d.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(
                    Helpers.flatten_dict(value, new_key, separator).items()
                )
            else:
                items.append((new_key, value))
        return dict(items)

    @staticmethod
    def generate_id(
        prefix: str = "",
        length: int = 12,
        method: str = "uuid",
    ) -> str:
        """Generate a unique identifier.

        Args:
            prefix: Optional prefix for the ID.
            length: Length of the random portion (uuid method only).
            method: Generation method ('uuid' or 'hash').

        Returns:
            Generated unique ID string.

        Raises:
            ValueError: If method is not 'uuid' or 'hash'.
        """
        if method == "uuid":
            random_part = uuid.uuid4().hex[:length]
        elif method == "hash":
            random_part = hashlib.sha256(
                f"{uuid.uuid4()}{time.time()}".encode()
            ).hexdigest()[:length]
        else:
            raise ValueError(f"method must be 'uuid' or 'hash', got '{method}'")

        return f"{prefix}{random_part}" if prefix else random_part

    @staticmethod
    def format_bytes(
        size_bytes: float,
        precision: int = 2,
    ) -> str:
        """Format a byte count as a human-readable string.

        Args:
            size_bytes: Number of bytes.
            precision: Decimal precision.

        Returns:
            Formatted string (e.g. '1.50 GiB').

        Raises:
            ValueError: If size_bytes is negative.
        """
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

        units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
        size = float(size_bytes)
        for unit in units:
            if abs(size) < 1024.0 or unit == units[-1]:
                return f"{size:.{precision}f} {unit}"
            size /= 1024.0

        return f"{size:.{precision}f} {units[-1]}"  # pragma: no cover

    @staticmethod
    def deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recursively merge two dictionaries.

        Values in *override* take precedence. Dicts are merged
        recursively; all other types are overwritten.

        Args:
            base: The base dictionary.
            override: The overriding dictionary.

        Returns:
            Merged dictionary.
        """
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Helpers.deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def safe_get(
        data: Dict[str, Any],
        key_path: str,
        default: Any = None,
        separator: str = ".",
    ) -> Any:
        """Safely retrieve a nested value by dot-separated key path.

        Args:
            data: Source dictionary.
            key_path: Dot-separated path (e.g. 'model.config.layers').
            default: Default value if path does not exist.
            separator: Path separator.

        Returns:
            The found value or *default*.
        """
        keys = key_path.split(separator)
        current: Any = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
