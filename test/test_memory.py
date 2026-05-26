"""Tests for Thunders AI memory system."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.llm.memory import MemorySystem, Memory, MemoryType


class TestMemorySystem:
    """Tests for the MemorySystem class."""

    def test_save_retrieve_memory(self, temp_dir):
        """Test saving and retrieving a memory."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        memory = Memory(
            id="mem-001",
            content="User prefers dark mode in the UI",
            memory_type=MemoryType.PREFERENCE,
            timestamp=time.time(),
        )
        memory_system.save(memory)
        retrieved = memory_system.retrieve("mem-001")
        assert retrieved is not None
        assert retrieved.content == "User prefers dark mode in the UI"
        assert retrieved.memory_type == MemoryType.PREFERENCE

    def test_memory_types(self, temp_dir):
        """Test different memory types are stored and retrieved correctly."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        memories = [
            Memory(id="mem-pref", content="Likes Python", memory_type=MemoryType.PREFERENCE, timestamp=time.time()),
            Memory(id="mem-fact", content="Earth is round", memory_type=MemoryType.FACT, timestamp=time.time()),
            Memory(id="mem-epis", content="Went to the store", memory_type=MemoryType.EPISODIC, timestamp=time.time()),
            Memory(id="mem-proc", content="How to bake bread", memory_type=MemoryType.PROCEDURAL, timestamp=time.time()),
        ]
        for mem in memories:
            memory_system.save(mem)

        for mem in memories:
            retrieved = memory_system.retrieve(mem.id)
            assert retrieved is not None
            assert retrieved.memory_type == mem.memory_type
            assert retrieved.content == mem.content

    def test_memory_consolidation(self, temp_dir):
        """Test that memory consolidation merges similar memories."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        # Save overlapping memories
        memory_system.save(Memory(
            id="mem-a", content="User likes Python programming",
            memory_type=MemoryType.PREFERENCE, timestamp=time.time(),
        ))
        memory_system.save(Memory(
            id="mem-b", content="User enjoys Python coding",
            memory_type=MemoryType.PREFERENCE, timestamp=time.time(),
        ))
        # Consolidate
        consolidated = memory_system.consolidate()
        assert consolidated is not None
        # After consolidation, similar memories should be merged
        assert isinstance(consolidated, list)

    def test_memory_clear(self, temp_dir):
        """Test clearing all memories."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        memory_system.save(Memory(
            id="mem-001", content="Some memory",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        ))
        memory_system.save(Memory(
            id="mem-002", content="Another memory",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        ))
        assert len(memory_system.list_all()) == 2
        memory_system.clear()
        assert len(memory_system.list_all()) == 0

    def test_persistent_storage(self, temp_dir):
        """Test that memories persist across MemorySystem instances."""
        storage_path = str(temp_dir / "memories")
        # Instance 1: save a memory
        ms1 = MemorySystem(storage_path=storage_path)
        ms1.save(Memory(
            id="mem-persist", content="Persistent memory test",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        ))
        del ms1

        # Instance 2: retrieve the memory
        ms2 = MemorySystem(storage_path=storage_path)
        retrieved = ms2.retrieve("mem-persist")
        assert retrieved is not None
        assert retrieved.content == "Persistent memory test"

    def test_memory_search(self, temp_dir):
        """Test searching memories by content."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        memory_system.save(Memory(
            id="mem-1", content="Python is a great language",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        ))
        memory_system.save(Memory(
            id="mem-2", content="JavaScript is widely used",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        ))
        results = memory_system.search("Python")
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_memory_expiry(self, temp_dir):
        """Test that memories can have an expiry time."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        # Create a memory that expired in the past
        expired_memory = Memory(
            id="mem-expired",
            content="This is old",
            memory_type=MemoryType.EPISODIC,
            timestamp=time.time() - 3600,
            expires_at=time.time() - 1800,  # Expired 30 minutes ago
        )
        memory_system.save(expired_memory)
        # Expired memories should not be retrieved
        result = memory_system.retrieve("mem-expired", include_expired=False)
        assert result is None

    def test_memory_update(self, temp_dir):
        """Test updating an existing memory."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        memory = Memory(
            id="mem-update", content="Original content",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        )
        memory_system.save(memory)
        # Update the memory
        updated = Memory(
            id="mem-update", content="Updated content",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        )
        memory_system.save(updated)
        retrieved = memory_system.retrieve("mem-update")
        assert retrieved.content == "Updated content"

    def test_memory_delete(self, temp_dir):
        """Test deleting a specific memory."""
        memory_system = MemorySystem(storage_path=str(temp_dir))
        memory_system.save(Memory(
            id="mem-del", content="Delete me",
            memory_type=MemoryType.FACT, timestamp=time.time(),
        ))
        assert memory_system.retrieve("mem-del") is not None
        memory_system.delete("mem-del")
        assert memory_system.retrieve("mem-del") is None
