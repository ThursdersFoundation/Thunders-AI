"""Caching utilities for Thunders AI.

Provides an LRU cache with TTL support, backed by memory
or disk storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class CacheEntry:
    """A single cache entry with metadata.

    Attributes:
        key: Cache key.
        value: Cached value.
        created_at: Time the entry was created.
        ttl: Time-to-live in seconds (None = no expiry).
    """

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.created_at: float = time.time()
        self.ttl = ttl
        self.access_count: int = 0
        self.last_accessed: float = self.created_at

    @property
    def is_expired(self) -> bool:
        """Check whether the entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Mark the entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


class Cache:
    """LRU cache with TTL support and disk persistence.

    Supports both in-memory and disk-backed caching with
    automatic eviction of expired or least-recently-used entries.

    Attributes:
        max_size: Maximum number of entries in memory.
        default_ttl: Default time-to-live in seconds.
    """

    def __init__(
        self,
        max_size: int = 1024,
        default_ttl: Optional[float] = None,
        disk_dir: Optional[str] = None,
        enable_disk: bool = False,
    ) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_disk = enable_disk
        self.disk_dir = Path(disk_dir) if disk_dir else None

        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

        if self.enable_disk and self.disk_dir:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Disk caching enabled: %s", self.disk_dir)
        else:
            logger.info("In-memory cache initialised: max_size=%d", max_size)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the cache.

        Args:
            key: Cache key.
            default: Value to return if key is missing or expired.

        Returns:
            Cached value or *default*.
        """
        # Check memory cache first
        if key in self._store:
            entry = self._store[key]
            if entry.is_expired:
                self._evict(key)
                self._misses += 1
                return default

            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

        # Check disk cache
        if self.enable_disk:
            disk_value = self._load_from_disk(key)
            if disk_value is not None:
                self._misses += 1  # Wasn't in memory
                # Promote to memory cache
                self._store[key] = CacheEntry(key=key, value=disk_value)
                self._enforce_size_limit()
                return disk_value

        self._misses += 1
        return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        persist_to_disk: bool = False,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds; uses default_ttl if None.
            persist_to_disk: Also write to disk cache.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl

        # Remove existing entry if present
        if key in self._store:
            del self._store[key]

        entry = CacheEntry(key=key, value=value, ttl=effective_ttl)
        self._store[key] = entry
        self._store.move_to_end(key)
        self._enforce_size_limit()

        if persist_to_disk and self.enable_disk and self.disk_dir:
            self._save_to_disk(key, value)

        logger.debug("Cache set: %s (ttl=%s)", key, effective_ttl)

    def delete(self, key: str) -> bool:
        """Remove an entry from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key was found and removed.
        """
        removed = False
        if key in self._store:
            del self._store[key]
            removed = True

        if self.enable_disk:
            disk_path = self._disk_path(key)
            if disk_path.exists():
                disk_path.unlink()
                removed = True

        if removed:
            logger.debug("Cache deleted: %s", key)
        return removed

    def clear(self, include_disk: bool = False) -> int:
        """Clear the cache.

        Args:
            include_disk: Also clear disk cache files.

        Returns:
            Number of entries removed.
        """
        count = len(self._store)
        self._store.clear()

        if include_disk and self.enable_disk and self.disk_dir:
            for f in self.disk_dir.iterdir():
                if f.is_file():
                    f.unlink()

        logger.info("Cache cleared: %d entries removed", count)
        return count

    def has(self, key: str) -> bool:
        """Check whether a key exists and is not expired.

        Args:
            key: Cache key to check.

        Returns:
            True if the key exists and is valid.
        """
        if key in self._store:
            entry = self._store[key]
            if not entry.is_expired:
                return True
            self._evict(key)

        if self.enable_disk:
            disk_path = self._disk_path(key)
            return disk_path.exists()

        return False

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, and hit rate.
        """
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "disk_enabled": self.enable_disk,
        }

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        expired_keys = [
            key for key, entry in self._store.items() if entry.is_expired
        ]
        for key in expired_keys:
            self._evict(key)

        if expired_keys:
            logger.info("Cleaned up %d expired entries", len(expired_keys))
        return len(expired_keys)

    # -- Internal helpers ---------------------------------------------------

    def _enforce_size_limit(self) -> None:
        """Evict the least recently used entry if the cache is over capacity."""
        while len(self._store) > self.max_size:
            oldest_key, _ = self._store.popitem(last=False)
            logger.debug("LRU evicted: %s", oldest_key)

    def _evict(self, key: str) -> None:
        """Remove a single entry by key."""
        self._store.pop(key, None)

    def _disk_path(self, key: str) -> Path:
        """Get the disk cache file path for a key."""
        if not self.disk_dir:
            raise ValueError("Disk caching not enabled")
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.disk_dir / f"{key_hash}.cache"

    def _save_to_disk(self, key: str, value: Any) -> None:
        """Persist a cache entry to disk."""
        if not self.disk_dir:
            return
        try:
            disk_path = self._disk_path(key)
            disk_path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:
            logger.error("Failed to save to disk: key=%s, error=%s", key, exc)

    def _load_from_disk(self, key: str) -> Any:
        """Load a cache entry from disk."""
        if not self.disk_dir:
            return None
        try:
            disk_path = self._disk_path(key)
            if disk_path.exists():
                return pickle.loads(disk_path.read_bytes())
        except Exception as exc:
            logger.error("Failed to load from disk: key=%s, error=%s", key, exc)
        return None
