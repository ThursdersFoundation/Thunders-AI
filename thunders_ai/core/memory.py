"""Memory system with episodic, semantic, and procedural memory types."""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Classification of memory sub-systems."""
    EPISODIC = "episodic"     # Event-based, temporal
    SEMANTIC = "semantic"     # Fact-based, knowledge
    PROCEDURAL = "procedural"  # Skill-based, how-to


class MemoryEntry:
    """A single memory record with metadata and relevance scoring."""

    __slots__ = ("id", "content", "memory_type", "timestamp", "metadata", "access_count")

    def __init__(
        self,
        entry_id: str,
        content: str,
        memory_type: MemoryType,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = entry_id
        self.content = content
        self.memory_type = memory_type
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        self.access_count = 0

    def touch(self) -> None:
        """Increment access count (used for relevance boosting)."""
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entry to a plain dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Deserialise an entry from a dictionary."""
        return cls(
            entry_id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


class MemorySystem:
    """Manages conversation history and long-term memory across memory types.

    Supports episodic, semantic, and procedural memory with relevance-based
    retrieval, memory consolidation, and persistent disk storage.

    Args:
        config: ThundersConfig instance.

    Example::

        mem = MemorySystem(config)
        mem.store("episodic", "User asked about weather")
        results = mem.retrieve("weather", top_k=3)
    """

    def __init__(self, config: ThundersConfig) -> None:
        self._config = config
        self._memories: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            mt: {} for mt in MemoryType
        }
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history: int = getattr(config, "max_conversation_history", 100)
        self._persist_path: Optional[str] = getattr(config, "memory_persist_path", None)

        if self._persist_path:
            self._load_from_disk()
        logger.info("MemorySystem initialized – max_history=%d", self._max_history)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        memory_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """Store a new memory entry.

        Args:
            memory_type: One of ``episodic``, ``semantic``, ``procedural``.
            content: The memory content string.
            metadata: Optional extra metadata.
            entry_id: Explicit ID; auto-generated if *None*.

        Returns:
            The entry ID.
        """
        mt = MemoryType(memory_type)
        if entry_id is None:
            entry_id = f"{mt.value}_{int(time.time() * 1000)}"
        entry = MemoryEntry(entry_id, content, mt, metadata=metadata)
        self._memories[mt][entry_id] = entry
        logger.debug("Stored %s memory: %s", mt.value, entry_id)
        return entry_id

    def add_to_conversation(self, role: str, content: str) -> None:
        """Append a message to the conversation history buffer.

        Args:
            role: Speaker role (e.g. ``user``, ``assistant``, ``system``).
            content: Message content.
        """
        self._conversation_history.append({"role": role, "content": content})
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[float, MemoryEntry]]:
        """Retrieve memories ranked by simple relevance scoring.

        Relevance is based on keyword overlap and access frequency.

        Args:
            query: Search query string.
            memory_type: Restrict to a single memory type.
            top_k: Number of results.

        Returns:
            List of ``(score, entry)`` tuples sorted by score descending.
        """
        query_tokens = set(query.lower().split())
        results: List[Tuple[float, MemoryEntry]] = []

        types_to_search = (
            [MemoryType(memory_type)] if memory_type else list(MemoryType)
        )
        for mt in types_to_search:
            for entry in self._memories[mt].values():
                content_tokens = set(entry.content.lower().split())
                overlap = len(query_tokens & content_tokens)
                if overlap == 0:
                    continue
                recency = 1.0 / (1.0 + (time.time() - entry.timestamp) / 3600)
                freq_bonus = 1.0 + 0.1 * min(entry.access_count, 10)
                score = (overlap / max(len(query_tokens), 1)) * recency * freq_bonus
                results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        for _, entry in results[:top_k]:
            entry.touch()
        return results[:top_k]

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Return the conversation history.

        Args:
            last_n: Return only the last *n* messages.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        if last_n:
            return self._conversation_history[-last_n:]
        return list(self._conversation_history)

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, max_age_hours: float = 168.0) -> int:
        """Remove stale or low-access episodic memories.

        Args:
            max_age_hours: Maximum age in hours before an episodic entry is
                eligible for removal (default 1 week).

        Returns:
            Number of entries removed.
        """
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for entry_id, entry in list(self._memories[MemoryType.EPISODIC].items()):
            if entry.timestamp < cutoff and entry.access_count < 2:
                del self._memories[MemoryType.EPISODIC][entry_id]
                removed += 1
        logger.info("Consolidation removed %d stale episodic memories.", removed)
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all memories to disk."""
        if not self._persist_path:
            logger.warning("No persist_path configured – skip saving.")
            return
        data: Dict[str, Any] = {
            "conversation_history": self._conversation_history,
            "memories": {
                mt.value: {eid: e.to_dict() for eid, e in entries.items()}
                for mt, entries in self._memories.items()
            },
        }
        Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self._persist_path).write_text(json.dumps(data, indent=2))
        logger.info("Memory persisted to %s", self._persist_path)

    def _load_from_disk(self) -> None:
        """Load persisted memories from disk."""
        if not self._persist_path or not Path(self._persist_path).exists():
            return
        data = json.loads(Path(self._persist_path).read_text())
        self._conversation_history = data.get("conversation_history", [])
        for mt_str, entries in data.get("memories", {}).items():
            mt = MemoryType(mt_str)
            for eid, edata in entries.items():
                self._memories[mt][eid] = MemoryEntry.from_dict(edata)
        logger.info("Loaded memories from %s", self._persist_path)
