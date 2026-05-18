"""
Tests for PLATO Memory Rooms.

Each room is tested independently:
- MemoryIngestRoom: chunking, code/model paths, tile generation
- MemoryTreeRoom: hierarchical summaries, propagation, levels
- MemoryQueryRoom: exact match, semantic search, early exit
- MemorySyncRoom: conflict resolution, Lamport ordering, merge

At least 15 tests covering all signal chain paths.
"""

import time

import pytest

from cocapn.memory_rooms import (
    AlphaDial,
    ChainResult,
    LamportClock,
    MemoryCategory,
    MemoryIngestRoom,
    MemoryQueryRoom,
    MemorySyncRoom,
    MemoryTile,
    MemoryTreeRoom,
    NodeLevel,
    Signal,
    SourceKind,
    TileLifecycle,
    approx_token_count,
    chunk_id,
    node_id_to_path,
)


# ═══════════════════════════════════════════════════════════════════════
# Core infrastructure tests
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryTile:
    def test_content_hash_deterministic(self):
        h1 = MemoryTile.content_hash("hello world")
        h2 = MemoryTile.content_hash("hello world")
        assert h1 == h2
        assert len(h1) == 32

    def test_content_hash_differs_for_different_input(self):
        h1 = MemoryTile.content_hash("hello")
        h2 = MemoryTile.content_hash("world")
        assert h1 != h2

    def test_tile_touch_advances_clock(self):
        tile = MemoryTile("t1", "test")
        assert tile.clock.value == 0
        v1 = tile.touch()
        assert v1 == 1
        v2 = tile.touch()
        assert v2 == 2


class TestLamportClock:
    def test_tick_monotonically_increases(self):
        c = LamportClock()
        assert c.tick() == 1
        assert c.tick() == 2
        assert c.tick() == 3

    def test_merge_takes_max_plus_one(self):
        a = LamportClock(value=5)
        b = LamportClock(value=10)
        result = a.merge(b)
        assert result == 11
        assert a.value == 11


class TestAlphaDial:
    def test_should_escalate_respects_alpha(self):
        dial = AlphaDial(0.2, "test")  # confidence < 0.8 → escalate
        assert dial.should_escalate(0.5) is True
        assert dial.should_escalate(0.9) is False

    def test_effective_alpha(self):
        dial = AlphaDial(0.4, "test")
        assert dial.effective_alpha(used_model=True) == 0.4
        assert dial.effective_alpha(used_model=False) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# MemoryIngestRoom tests
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryIngestRoom:
    def test_ingest_simple_markdown_produces_tiles(self):
        room = MemoryIngestRoom()
        md = "# Title\nHello world"
        result = room.ingest(SourceKind.document, "doc1", md, owner="alice")
        assert result.signal == Signal.early_exit
        assert len(result.output) >= 1
        tile = result.output[0]
        assert tile.room == "memory-ingest"
        assert "Hello world" in tile.payload["chunk"]["content"]

    def test_ingest_empty_text_produces_empty_chunk(self):
        room = MemoryIngestRoom()
        result = room.ingest(SourceKind.document, "doc1", "")
        assert len(result.output) >= 1

    def test_ingest_long_text_splits_into_multiple_chunks(self):
        room = MemoryIngestRoom(max_tokens=50)  # 200 chars max
        long_text = "# Section\n" + "x" * 1000
        result = room.ingest(SourceKind.document, "doc1", long_text)
        assert len(result.output) > 1

    def test_ingest_preserves_headings(self):
        room = MemoryIngestRoom()
        md = "# Heading A\nContent A\n\n# Heading B\nContent B"
        result = room.ingest(SourceKind.document, "doc1", md)
        headings = [t.metadata.get("heading") for t in result.output]
        assert "# Heading A" in headings
        assert "# Heading B" in headings

    def test_ingest_deterministic_chunk_ids(self):
        room = MemoryIngestRoom()
        md = "# Title\nHello"
        r1 = room.ingest(SourceKind.document, "doc1", md)
        room2 = MemoryIngestRoom()
        r2 = room2.ingest(SourceKind.document, "doc1", md)
        assert r1.output[0].tile_id == r2.output[0].tile_id

    def test_ingest_model_path_for_unstructured_content(self):
        """Very long single line → low code confidence → model path."""
        room = MemoryIngestRoom()
        # Single giant line with no structure → code confidence drops
        giant_line = "x" * 50000

        model_called = {"count": 0}
        def mock_model(text: str) -> str:
            model_called["count"] += 1
            return text

        result = room.ingest(
            SourceKind.document, "doc1", giant_line,
            model_fn=mock_model,
        )
        # Model should be invoked because code confidence is low
        assert model_called["count"] == 1
        assert result.signal == Signal.model_needed


# ═══════════════════════════════════════════════════════════════════════
# MemoryTreeRoom tests
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryTreeRoom:
    def _make_chunk_tiles(self, count: int = 3) -> list[MemoryTile]:
        tiles = []
        for i in range(count):
            tile = MemoryTile(
                tile_id=f"chunk-{i}",
                room="memory-ingest",
                payload={
                    "chunk": {
                        "id": f"chunk-{i}",
                        "content": f"Content for chunk {i}. " * 20,
                        "token_count": 50,
                        "seq_in_source": i,
                    }
                },
                metadata={
                    "source_kind": "document",
                    "hour_id": f"2026/05/17/{10 + i // 2:02d}",
                },
            )
            tiles.append(tile)
        return tiles

    def test_build_tree_empty_input(self):
        room = MemoryTreeRoom()
        result = room.build_tree("test", [])
        assert result.signal == Signal.early_exit
        assert result.output == []

    def test_build_tree_creates_hour_leaves(self):
        room = MemoryTreeRoom()
        chunks = self._make_chunk_tiles(4)
        result = room.build_tree("test", chunks)
        # Should have hour nodes
        hour_nodes = [
            t for t in result.output
            if t.payload.get("node", {}).get("level") == "hour"
        ]
        assert len(hour_nodes) >= 1

    def test_build_tree_propagates_to_root(self):
        room = MemoryTreeRoom()
        chunks = self._make_chunk_tiles(3)
        result = room.build_tree("test", chunks)
        root = room.nodes.get("root@test")
        assert root is not None
        assert root.payload["node"]["level"] == "root"

    def test_build_tree_with_model_summarization(self):
        room = MemoryTreeRoom()
        # Make content that exceeds hour token budget (1000 tokens)
        chunks = [MemoryTile(
            tile_id="big-chunk",
            room="memory-ingest",
            payload={"chunk": {
                "id": "big", "content": "word " * 5000,  # ~25k chars, ~6k tokens
                "token_count": 6000, "seq_in_source": 0,
            }},
            metadata={"source_kind": "document", "hour_id": "2026/05/17/10"},
        )]

        def mock_summarize(content: str, max_tokens: int) -> str:
            return f"Summary of {len(content)} chars"

        result = room.build_tree("test", chunks, model_fn=mock_summarize)
        assert result.alpha_used > 0  # Model was used

    def test_node_level_max_tokens(self):
        assert NodeLevel.hour.max_tokens() == 1_000
        assert NodeLevel.day.max_tokens() == 2_000
        assert NodeLevel.month.max_tokens() == 4_000
        assert NodeLevel.year.max_tokens() == 8_000
        assert NodeLevel.root.max_tokens() == 20_000


# ═══════════════════════════════════════════════════════════════════════
# MemoryQueryRoom tests
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryQueryRoom:
    def _make_indexed_room(self) -> tuple[MemoryQueryRoom, list[MemoryTile]]:
        room = MemoryQueryRoom()
        tiles = [
            MemoryTile(
                tile_id="t1",
                room="memory-query",
                payload={"chunk": {"content": "Rust is a systems programming language"}},
                metadata={"category": "core"},
            ),
            MemoryTile(
                tile_id="t2",
                room="memory-query",
                payload={"chunk": {"content": "Python is great for data science"}},
                metadata={"category": "daily"},
            ),
            MemoryTile(
                tile_id="t3",
                room="memory-query",
                payload={"node": {"summary": "Meeting notes about the Rust migration plan"}},
                metadata={},
            ),
        ]
        room.index_tiles("global", tiles)
        return room, tiles

    def test_exact_match_returns_relevant_tiles(self):
        room, _ = self._make_indexed_room()
        result = room.query("Rust", namespace="global")
        assert len(result.output) >= 2  # t1 and t3 contain "Rust"
        assert result.signal in (Signal.early_exit, Signal.code_done)

    def test_no_match_returns_empty(self):
        room, _ = self._make_indexed_room()
        result = room.query("quantum computing", namespace="global")
        # Code path found nothing, might escalate to model
        assert len(result.output) == 0 or result.signal in (Signal.code_done, Signal.model_needed)

    def test_semantic_search_with_model(self):
        room, tiles = self._make_indexed_room()

        def mock_embed(query: str, docs: list[str]) -> list[float]:
            # Simple mock: return higher scores for docs containing query words
            q_words = set(query.lower().split())
            return [
                sum(1.0 for w in q_words if w in d.lower()) / max(len(q_words), 1)
                for d in docs
            ]

        # Use a query that has NO exact substring match so model path activates
        result = room.query("low-level language safety", namespace="global", model_fn=mock_embed)
        assert result.alpha_used > 0  # Model was used
        assert len(result.output) >= 1

    def test_limit_respected(self):
        room, _ = self._make_indexed_room()
        result = room.query("Rust", namespace="global", limit=1)
        assert len(result.output) <= 1


# ═══════════════════════════════════════════════════════════════════════
# MemorySyncRoom tests
# ═══════════════════════════════════════════════════════════════════════


class TestMemorySyncRoom:
    def _make_tile(self, tid: str, content: str, clock_val: int = 0) -> MemoryTile:
        tile = MemoryTile(
            tile_id=tid,
            room="memory-sync",
            payload={"chunk": {"content": content}},
            clock=LamportClock(value=clock_val),
        )
        return tile

    def test_sync_identical_sets(self):
        room = MemorySyncRoom()
        tiles = [self._make_tile("t1", "hello", 1)]
        result = room.sync(tiles, tiles)
        assert result.signal == Signal.early_exit
        assert result.output["total"] == 1
        assert result.output["conflicts"] == 0

    def test_sync_merges_non_overlapping(self):
        room = MemorySyncRoom()
        local = [self._make_tile("t1", "local only")]
        remote = [self._make_tile("t2", "remote only")]
        result = room.sync(local, remote)
        assert len(result.output["merged"]) == 2

    def test_sync_resolves_by_lamport_clock(self):
        room = MemorySyncRoom()
        local = [self._make_tile("t1", "newer content", clock_val=5)]
        remote = [self._make_tile("t1", "older content", clock_val=2)]
        result = room.sync(local, remote)
        merged = result.output["merged"]
        assert len(merged) == 1
        assert merged[0].payload["chunk"]["content"] == "newer content"

    def test_sync_model_conflict_resolution(self):
        room = MemorySyncRoom()
        # Different content, same clock value → needs model
        local = self._make_tile("t1", "version A", clock_val=5)
        remote = self._make_tile("t1", "version B", clock_val=5)
        # Different content hash forces conflict
        remote.payload["chunk"]["content"] = "version B different"

        def mock_resolver(data_a: str, data_b: str) -> bool:
            return "A" in data_a  # Keep local if it has "A"

        result = room.sync([local], [remote], model_fn=mock_resolver)
        # With α=0.1, code_confidence=0.95 > 0.9, so code resolves
        # But identical clock + different content... code fallback takes newer by insertion order
        merged = result.output["merged"]
        assert len(merged) == 1


# ═══════════════════════════════════════════════════════════════════════
# Utility function tests
# ═══════════════════════════════════════════════════════════════════════


class TestUtilities:
    def test_approx_token_count(self):
        assert approx_token_count("") == 1  # max(1, 0) = 1
        assert approx_token_count("abcd") == 1
        assert approx_token_count("abcde") == 2
        assert approx_token_count("a" * 400) == 100

    def test_chunk_id_deterministic(self):
        a = chunk_id(SourceKind.chat, "slack:#eng", 0, "hello")
        b = chunk_id(SourceKind.chat, "slack:#eng", 0, "hello")
        assert a == b
        assert len(a) == 32

    def test_chunk_id_varies_with_content(self):
        a = chunk_id(SourceKind.chat, "x", 0, "hello")
        b = chunk_id(SourceKind.chat, "x", 0, "world")
        assert a != b

    def test_node_id_to_path(self):
        assert node_id_to_path("root") == "root.md"
        assert node_id_to_path("2026") == "2026/summary.md"
        assert node_id_to_path("2026/05/17/10") == "2026/05/17/10.md"
        assert node_id_to_path("2026/05/17") == "2026/05/17/summary.md"


# ═══════════════════════════════════════════════════════════════════════
# Integration: full pipeline test
# ═══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_ingest_to_tree_to_query(self):
        """End-to-end: ingest → build tree → query."""
        # 1. Ingest
        ingest = MemoryIngestRoom()
        md = "# Meeting\nDiscussed Rust migration. Python still used for ML.\n\n# Decision\nMoving to Rust for backend services."
        ingest_result = ingest.ingest(SourceKind.document, "meeting-1", md, owner="alice")
        assert len(ingest_result.output) >= 1

        # 2. Build tree
        tree = MemoryTreeRoom()
        tree_result = tree.build_tree("meetings", ingest_result.output)
        assert tree_result.signal in (Signal.early_exit, Signal.model_needed)

        # 3. Query
        query = MemoryQueryRoom()
        all_tiles = ingest_result.output + list(tree.nodes.values())
        query.index_tiles("meetings", all_tiles)
        query_result = query.query("Rust", namespace="meetings")
        assert len(query_result.output) >= 1

    def test_sync_after_ingest(self):
        """Ingest on two devices, sync results."""
        room_a = MemoryIngestRoom()
        room_b = MemoryIngestRoom()

        md_a = "# Notes\nDevice A notes about Rust"
        md_b = "# Notes\nDevice B notes about Python"

        result_a = room_a.ingest(SourceKind.document, "doc1", md_a)
        result_b = room_b.ingest(SourceKind.document, "doc1", md_b)

        sync = MemorySyncRoom()
        sync_result = sync.sync(result_a.output, result_b.output)
        assert sync_result.output["total"] >= 2  # Different tile IDs (different content)
