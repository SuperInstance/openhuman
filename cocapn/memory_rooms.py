"""
PLATO Memory Rooms for OpenHuman's memory system.

Maps OpenHuman's memory/tree/chunker/summarizer architecture to PLATO rooms
using the spreader-tool signal chain pattern: tiles, α dials, early exit.

Rooms:
- MemoryIngestRoom  (α=0.2): chunk documents, code splits, model handles novel formats
- MemoryTreeRoom    (α=0.3): hierarchical summaries, code for fixed schemas, model for novel
- MemoryQueryRoom   (α=0.4): retrieve memories, code does exact match, model does semantic
- MemorySyncRoom    (α=0.1): cross-device sync, pure code with micro-model conflict detection

Each room is independently importable and testable.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════
# Core tile infrastructure (mirrors plato-types)
# ═══════════════════════════════════════════════════════════════════════


class TileLifecycle(str, Enum):
    active = "active"
    superseded = "superseded"
    retracted = "retracted"


class Signal(str, Enum):
    """Exit signals from the signal chain."""
    code_done = "code_done"        # Code path resolved it
    model_needed = "model_needed"  # Escalate to model
    early_exit = "early_exit"      # Resolved before model needed
    fallback = "fallback"          # Model failed, code fallback


@dataclass
class LamportClock:
    """Logical clock for causal ordering across rooms."""
    value: int = 0

    def tick(self) -> int:
        self.value += 1
        return self.value

    def merge(self, other: "LamportClock") -> int:
        self.value = max(self.value, other.value) + 1
        return self.value


@dataclass
class MemoryTile:
    """
    A tile carrying memory context through the signal chain.

    Maps OpenHuman's data structures:
    - Chunk (tree/types.rs) → tile.payload['chunk']
    - MemoryEntry (traits.rs) → tile.payload['entry']
    - TreeNode (tree_summarizer/types.rs) → tile.payload['node']
    - SourceKind / DataSource → tile.metadata['source_kind']
    """
    tile_id: str
    room: str
    lifecycle: TileLifecycle = TileLifecycle.active
    clock: LamportClock = field(default_factory=LamportClock)
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def content_hash(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def touch(self) -> int:
        return self.clock.tick()


# ═══════════════════════════════════════════════════════════════════════
# Signal chain primitives
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ChainResult:
    """Result of a signal chain pass through a room."""
    signal: Signal
    tile: MemoryTile
    output: Any = None
    alpha_used: float = 0.0  # how much model was needed (0=pure code, 1=pure model)
    latency_ms: float = 0.0


class AlphaDial:
    """
    Controls code-vs-model balance in a room.

    α=0.0 → pure code (deterministic, fast)
    α=1.0 → pure model (flexible, slow)

    The dial governs when the signal chain escalates:
    - Code path runs first
    - If confidence < (1 - α), escalate to model
    - Early exit when code resolves with high confidence
    """

    def __init__(self, alpha: float, name: str = ""):
        self.alpha = max(0.0, min(1.0, alpha))
        self.name = name

    def should_escalate(self, code_confidence: float) -> bool:
        """Return True if the model path should be activated."""
        return code_confidence < (1.0 - self.alpha)

    def effective_alpha(self, used_model: bool) -> float:
        """Report actual alpha after chain completes."""
        return self.alpha if used_model else 0.0


# ═══════════════════════════════════════════════════════════════════════
# OpenHuman data structure mappings
# ═══════════════════════════════════════════════════════════════════════


class SourceKind(str, Enum):
    """Mirrors tree/types.rs SourceKind."""
    chat = "chat"
    email = "email"
    document = "document"


class MemoryCategory(str, Enum):
    """Mirrors traits.rs MemoryCategory."""
    core = "core"
    daily = "daily"
    conversation = "conversation"


class NodeLevel(str, Enum):
    """Mirrors tree_summarizer/types.rs NodeLevel."""
    root = "root"
    year = "year"
    month = "month"
    day = "day"
    hour = "hour"

    def max_tokens(self) -> int:
        limits = {
            "root": 20_000,
            "year": 8_000,
            "month": 4_000,
            "day": 2_000,
            "hour": 1_000,
        }
        return limits[self.value]


def approx_token_count(text: str) -> int:
    """OpenHuman's token heuristic: ~4 chars per token."""
    return max(1, (len(text) + 3) // 4)


def chunk_id(source_kind: SourceKind, source_id: str, seq: int, content: str) -> str:
    """
    Deterministic chunk ID matching tree/types.rs::chunk_id.
    sha256(kind | \0 | source_id | \0 | seq | \0 | content)[:32]
    """
    h = hashlib.sha256()
    h.update(source_kind.value.encode())
    h.update(b"\x00")
    h.update(source_id.encode())
    h.update(b"\x00")
    h.update(seq.to_bytes(4, "big"))
    h.update(b"\x00")
    h.update(content.encode())
    return h.hexdigest()[:32]


def node_id_to_path(node_id: str) -> str:
    """Mirrors tree_summarizer/types.rs node_id_to_path."""
    if node_id == "root":
        return "root.md"
    parts = node_id.split("/")
    if len(parts) == 4:  # hour leaf
        return f"{node_id}.md"
    return f"{node_id}/summary.md"


# ═══════════════════════════════════════════════════════════════════════
# MemoryIngestRoom (α=0.2)
# ═══════════════════════════════════════════════════════════════════════


class MemoryIngestRoom:
    """
    Chunks documents into tiles.

    Signal chain:
    1. Code path: markdown chunker (heading/paragraph/line splitting)
       → high confidence for standard formats
    2. Model path: novel format detection (PDF, binary, non-standard)
       → low α means code handles most cases

    α=0.2 → 80% confidence threshold for code path.
    """

    def __init__(self, max_tokens: int = 3000):
        self.dial = AlphaDial(0.2, "memory-ingest")
        self.max_tokens = max_tokens
        self.chunks: list[MemoryTile] = []

    def ingest(
        self,
        source_kind: SourceKind,
        source_id: str,
        markdown: str,
        owner: str = "",
        tags: list[str] | None = None,
        model_fn: Callable[[str], str] | None = None,
    ) -> ChainResult:
        """
        Ingest a document through the signal chain.

        Code path: split markdown into chunks using heading/paragraph logic.
        Model path: if code confidence is low (novel format), invoke model_fn.

        Returns a ChainResult with all chunks as tiles.
        """
        t0 = time.monotonic()
        tags = tags or []

        # ── Code path: standard markdown chunking ──
        code_confidence = self._assess_code_confidence(markdown, source_kind)
        raw_chunks = self._chunk_markdown(markdown)
        used_model = False

        # ── Early exit if code is confident ──
        if not self.dial.should_escalate(code_confidence):
            tiles = self._chunks_to_tiles(raw_chunks, source_kind, source_id, owner, tags)
            self.chunks.extend(tiles)
            elapsed = (time.monotonic() - t0) * 1000
            return ChainResult(
                signal=Signal.early_exit,
                tile=tiles[0] if tiles else MemoryTile("empty", self.dial.name),
                output=tiles,
                alpha_used=0.0,
                latency_ms=elapsed,
            )

        # ── Model path: novel format handling ──
        if model_fn is not None:
            enhanced = model_fn(markdown)
            raw_chunks = self._chunk_markdown(enhanced)
            used_model = True
        else:
            # Fallback: still use code chunks but note model was needed
            used_model = False

        tiles = self._chunks_to_tiles(raw_chunks, source_kind, source_id, owner, tags)
        self.chunks.extend(tiles)
        elapsed = (time.monotonic() - t0) * 1000

        return ChainResult(
            signal=Signal.model_needed if used_model else Signal.early_exit,
            tile=tiles[0] if tiles else MemoryTile("empty", self.dial.name),
            output=tiles,
            alpha_used=self.dial.effective_alpha(used_model),
            latency_ms=elapsed,
        )

    def _assess_code_confidence(self, text: str, source_kind: SourceKind) -> float:
        """
        How confident is the code path in handling this input?

        High confidence: well-structured markdown with headings.
        Low confidence: binary-ish, very long lines, no structure.
        """
        if not text.strip():
            return 0.5

        has_headings = any(
            line.startswith("#") for line in text.split("\n")
        )
        has_paragraphs = "\n\n" in text
        line_count = text.count("\n")
        max_line_len = max((len(l) for l in text.split("\n")), default=0)

        confidence = 0.5
        if has_headings:
            confidence += 0.2
        if has_paragraphs:
            confidence += 0.15
        if line_count > 5:
            confidence += 0.1
        if max_line_len > 10000:
            confidence -= 0.2  # very long single lines → less confident
        if source_kind == SourceKind.document:
            confidence += 0.05  # documents are our sweet spot

        return min(1.0, max(0.0, confidence))

    def _chunk_markdown(self, text: str) -> list[dict]:
        """
        Split markdown into chunks. Mirrors chunker.rs logic:
        heading → paragraph → line → char boundary hierarchy.
        """
        if not text.strip():
            return [{"index": 0, "content": "", "heading": None}]

        max_chars = self.max_tokens * 4
        sections = self._split_on_headings(text)
        chunks: list[dict] = []

        for heading, body in sections:
            full = f"{heading}\n{body}" if heading else body

            if len(full) <= max_chars:
                chunks.append({"index": len(chunks), "content": full.strip(), "heading": heading})
            else:
                # Split on paragraphs
                paragraphs = [p for p in body.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [body]

                current = f"{heading}\n" if heading else ""
                for para in paragraphs:
                    if len(current) + len(para) > max_chars and current.strip():
                        chunks.append({"index": len(chunks), "content": current.strip(), "heading": heading})
                        current = f"{heading}\n" if heading else ""

                    if len(para) > max_chars:
                        if current.strip():
                            chunks.append({"index": len(chunks), "content": current.strip(), "heading": heading})
                            current = f"{heading}\n" if heading else ""
                        # Hard split on char boundaries
                        for i in range(0, len(para), max_chars):
                            piece = para[i:i + max_chars]
                            if piece.strip():
                                chunks.append({"index": len(chunks), "content": piece.strip(), "heading": heading})
                    else:
                        current += para + "\n"

                if current.strip():
                    chunks.append({"index": len(chunks), "content": current.strip(), "heading": heading})

        # Re-index
        for i, c in enumerate(chunks):
            c["index"] = i

        return chunks

    def _split_on_headings(self, text: str) -> list[tuple[str | None, str]]:
        """Split on ATX headings (# through ######)."""
        sections: list[tuple[str | None, str]] = []
        current_heading: str | None = None
        current_body = ""

        for line in text.split("\n"):
            if self._is_atx_heading(line):
                if current_body.strip() or current_heading is not None:
                    sections.append((current_heading, current_body))
                current_heading = line
                current_body = ""
            else:
                current_body += line + "\n"

        if current_body.strip() or current_heading is not None:
            sections.append((current_heading, current_body))

        return sections

    @staticmethod
    def _is_atx_heading(line: str) -> bool:
        prefixes = ["# ", "## ", "### ", "#### ", "##### ", "###### "]
        return any(line.startswith(p) for p in prefixes)

    def _chunks_to_tiles(
        self,
        chunks: list[dict],
        source_kind: SourceKind,
        source_id: str,
        owner: str,
        tags: list[str],
    ) -> list[MemoryTile]:
        tiles = []
        for chunk in chunks:
            content = chunk["content"]
            seq = chunk["index"]
            cid = chunk_id(source_kind, source_id, seq, content)
            tile = MemoryTile(
                tile_id=cid,
                room="memory-ingest",
                payload={
                    "chunk": {
                        "id": cid,
                        "content": content,
                        "token_count": approx_token_count(content),
                        "seq_in_source": seq,
                        "partial_message": False,
                    }
                },
                metadata={
                    "source_kind": source_kind.value,
                    "source_id": source_id,
                    "owner": owner,
                    "tags": tags,
                    "heading": chunk.get("heading"),
                },
            )
            tiles.append(tile)
        return tiles


# ═══════════════════════════════════════════════════════════════════════
# MemoryTreeRoom (α=0.3)
# ═══════════════════════════════════════════════════════════════════════


class MemoryTreeRoom:
    """
    Builds hierarchical summaries from chunks.

    Signal chain:
    1. Code path: fixed schema summarization (merge children, propagate up)
       → confident for known levels (hour/day/month/year/root)
    2. Model path: novel content that doesn't fit the fixed schema
       → α=0.3 means model handles ~30% of cases

    Mirrors tree_summarizer/engine.rs: run_summarization + propagate_node.
    """

    def __init__(self):
        self.dial = AlphaDial(0.3, "memory-tree")
        self.nodes: dict[str, MemoryTile] = {}

    def build_tree(
        self,
        namespace: str,
        chunks: list[MemoryTile],
        model_fn: Callable[[str, int], str] | None = None,
    ) -> ChainResult:
        """
        Build summary tree from ingested chunks.

        1. Group chunks by hour (from metadata timestamp)
        2. Summarize each hour into a leaf node
        3. Propagate summaries upward: hour → day → month → year → root
        """
        t0 = time.monotonic()

        if not chunks:
            return ChainResult(
                signal=Signal.early_exit,
                tile=MemoryTile("empty-tree", self.dial.name),
                output=[],
                alpha_used=0.0,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # Group by hour
        hour_groups: dict[str, list[MemoryTile]] = {}
        for chunk in chunks:
            hour_id = chunk.metadata.get("hour_id", "unknown/00/00/00")
            hour_groups.setdefault(hour_id, []).append(chunk)

        used_model = False

        # Build hour leaves
        for hour_id, group in hour_groups.items():
            combined = "\n\n---\n\n".join(
                t.payload.get("chunk", {}).get("content", "") for t in group
            )
            leaf = self._summarize_node(
                namespace, hour_id, NodeLevel.hour, combined, model_fn
            )
            if leaf.get("used_model"):
                used_model = True

        # Propagate upward
        propagation_needed: set[str] = set()
        for hour_id in hour_groups:
            parts = hour_id.split("/")
            if len(parts) == 4:
                propagation_needed.add("/".join(parts[:3]))  # day
                propagation_needed.add("/".join(parts[:2]))  # month
                propagation_needed.add(parts[0])             # year

        # Process day → month → year → root (bottom-up)
        levels = [
            (NodeLevel.day, lambda nid: "/".join(nid.split("/")[:3]) if len(nid.split("/")) >= 3 else None),
            (NodeLevel.month, lambda nid: "/".join(nid.split("/")[:2]) if len(nid.split("/")) >= 2 else None),
            (NodeLevel.year, lambda nid: nid.split("/")[0] if nid.split("/") else None),
        ]

        for level, _ in levels:
            for node_id in list(propagation_needed):
                existing_level = self._level_from_node_id(node_id)
                if existing_level == level:
                    self._propagate(namespace, node_id, level, model_fn)

        # Root
        if propagation_needed:
            r = self._propagate(namespace, "root", NodeLevel.root, model_fn)
            if r.get("used_model"):
                used_model = True

        elapsed = (time.monotonic() - t0) * 1000
        root_tile = self.nodes.get(f"root@{namespace}", MemoryTile("no-root", self.dial.name))

        return ChainResult(
            signal=Signal.model_needed if used_model else Signal.early_exit,
            tile=root_tile,
            output=list(self.nodes.values()),
            alpha_used=self.dial.effective_alpha(used_model),
            latency_ms=elapsed,
        )

    def _summarize_node(
        self,
        namespace: str,
        node_id: str,
        level: NodeLevel,
        content: str,
        model_fn: Callable[[str, int], str] | None,
    ) -> dict:
        """Summarize content for a single node. Code first, model if needed."""
        max_tokens = level.max_tokens()
        content_tokens = approx_token_count(content)

        # Code path: if content fits, use it directly
        if content_tokens <= max_tokens:
            summary = content
            used_model = False
        elif model_fn is not None:
            # Model path: summarize to fit
            summary = model_fn(content, max_tokens)
            used_model = True
        else:
            # Fallback: truncate to fit
            max_chars = max_tokens * 4
            summary = content[:max_chars]
            used_model = False

        tile = MemoryTile(
            tile_id=MemoryTile.content_hash(f"{namespace}/{node_id}/{summary[:64]}"),
            room="memory-tree",
            payload={
                "node": {
                    "node_id": node_id,
                    "namespace": namespace,
                    "level": level.value,
                    "summary": summary,
                    "token_count": approx_token_count(summary),
                }
            },
            metadata={"namespace": namespace, "level": level.value},
        )
        self.nodes[f"{node_id}@{namespace}"] = tile
        return {"used_model": used_model, "tile": tile}

    def _propagate(
        self,
        namespace: str,
        node_id: str,
        level: NodeLevel,
        model_fn: Callable[[str, int], str] | None,
    ) -> dict:
        """Propagate summary from children (mirrors engine.rs propagate_node)."""
        # Collect children
        children: list[MemoryTile] = []
        prefix = node_id if node_id != "root" else ""
        for key, tile in self.nodes.items():
            nid = tile.payload.get("node", {}).get("node_id", "")
            ns = tile.metadata.get("namespace", "")
            if ns != namespace:
                continue
            if node_id == "root":
                if self._level_from_node_id(nid) == NodeLevel.year:
                    children.append(tile)
            else:
                parent = self._derive_parent_id(nid)
                if parent == node_id and nid != node_id:
                    children.append(tile)

        if not children:
            return {"used_model": False}

        combined = "\n\n---\n\n".join(
            f"## {t.payload['node']['node_id']} ({t.payload['node']['level']})\n\n"
            f"{t.payload['node']['summary']}"
            for t in children
        )

        return self._summarize_node(namespace, node_id, level, combined, model_fn)

    @staticmethod
    def _level_from_node_id(node_id: str) -> NodeLevel:
        if node_id == "root":
            return NodeLevel.root
        slashes = node_id.count("/")
        return [NodeLevel.year, NodeLevel.month, NodeLevel.day, NodeLevel.hour][
            min(slashes, 3)
        ]

    @staticmethod
    def _derive_parent_id(node_id: str) -> str | None:
        if node_id == "root":
            return None
        idx = node_id.rfind("/")
        return node_id[:idx] if idx >= 0 else "root"


# ═══════════════════════════════════════════════════════════════════════
# MemoryQueryRoom (α=0.4)
# ═══════════════════════════════════════════════════════════════════════


class MemoryQueryRoom:
    """
    Retrieve memories from the tile store.

    Signal chain:
    1. Code path: exact match (keyword, ID lookup, namespace filter)
       → fast, deterministic
    2. Model path: semantic search (embedding similarity, fuzzy match)
       → α=0.4 means model handles ~40% of queries

    Mirrors memory/tree/retrieval/ logic + traits.rs Memory::recall.
    """

    def __init__(self):
        self.dial = AlphaDial(0.4, "memory-query")
        self.index: dict[str, list[MemoryTile]] = {}  # namespace → tiles

    def index_tiles(self, namespace: str, tiles: list[MemoryTile]) -> None:
        """Load tiles into the query index."""
        self.index[namespace] = tiles

    def query(
        self,
        query: str,
        namespace: str = "global",
        limit: int = 10,
        category: MemoryCategory | None = None,
        model_fn: Callable[[str, list[str]], list[float]] | None = None,
    ) -> ChainResult:
        """
        Query memories through the signal chain.

        1. Code path: exact/substring match against tile content
        2. Model path: semantic similarity via embeddings
        """
        t0 = time.monotonic()
        tiles = self.index.get(namespace, [])

        # ── Code path: exact match ──
        exact_hits = self._exact_match(query, tiles, category)
        code_confidence = 0.9 if exact_hits else 0.3

        if not self.dial.should_escalate(code_confidence):
            results = exact_hits[:limit]
            elapsed = (time.monotonic() - t0) * 1000
            return ChainResult(
                signal=Signal.early_exit if results else Signal.code_done,
                tile=results[0] if results else MemoryTile("no-match", self.dial.name),
                output=results,
                alpha_used=0.0,
                latency_ms=elapsed,
            )

        # ── Model path: semantic search ──
        used_model = False
        if model_fn is not None and tiles:
            contents = [
                t.payload.get("chunk", {}).get("content", "")
                or t.payload.get("node", {}).get("summary", "")
                for t in tiles
            ]
            scores = model_fn(query, contents)
            scored = list(zip(tiles, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [t for t, s in scored[:limit]]
            used_model = True
        else:
            # Fallback: broader substring match
            q_lower = query.lower()
            results = [
                t for t in tiles
                if q_lower in self._tile_text(t).lower()
            ][:limit]

        elapsed = (time.monotonic() - t0) * 1000
        return ChainResult(
            signal=Signal.model_needed if used_model else Signal.fallback,
            tile=results[0] if results else MemoryTile("no-match", self.dial.name),
            output=results,
            alpha_used=self.dial.effective_alpha(used_model),
            latency_ms=elapsed,
        )

    def _exact_match(
        self,
        query: str,
        tiles: list[MemoryTile],
        category: MemoryCategory | None,
    ) -> list[MemoryTile]:
        """Exact keyword match against tile content."""
        results = []
        for tile in tiles:
            if category and tile.metadata.get("category") != category.value:
                continue
            text = self._tile_text(tile)
            if query.lower() in text.lower():
                results.append(tile)
        return results

    @staticmethod
    def _tile_text(tile: MemoryTile) -> str:
        """Extract searchable text from a tile."""
        if "chunk" in tile.payload:
            return tile.payload["chunk"].get("content", "")
        if "node" in tile.payload:
            return tile.payload["node"].get("summary", "")
        return ""


# ═══════════════════════════════════════════════════════════════════════
# MemorySyncRoom (α=0.1)
# ═══════════════════════════════════════════════════════════════════════


class MemorySyncRoom:
    """
    Sync memory tiles across devices.

    Signal chain:
    1. Code path: hash-based conflict detection, merge strategies
       → handles 99% of cases deterministically
    2. Model path: micro-model for semantic conflict detection (rare)
       → α=0.1 means almost never needed

    Pure code with a tiny model escape hatch for genuine conflicts.
    """

    def __init__(self):
        self.dial = AlphaDial(0.1, "memory-sync")
        self.local_state: dict[str, MemoryTile] = {}
        self.remote_state: dict[str, MemoryTile] = {}

    def sync(
        self,
        local_tiles: list[MemoryTile],
        remote_tiles: list[MemoryTile],
        model_fn: Callable[[str, str], bool] | None = None,
    ) -> ChainResult:
        """
        Sync local and remote tile sets.

        Strategy:
        1. Hash comparison → detect conflicts
        2. Lamport clock → resolve most conflicts (newer wins)
        3. Model path → semantic conflict resolution (rare)
        """
        t0 = time.monotonic()

        # Index by tile_id
        self.local_state = {t.tile_id: t for t in local_tiles}
        self.remote_state = {t.tile_id: t for t in remote_tiles}

        all_ids = set(self.local_state.keys()) | set(self.remote_state.keys())
        merged: list[MemoryTile] = []
        conflicts: list[tuple[MemoryTile, MemoryTile]] = []
        used_model = False

        for tid in all_ids:
            local = self.local_state.get(tid)
            remote = self.remote_state.get(tid)

            if local and not remote:
                merged.append(local)
            elif remote and not local:
                merged.append(remote)
            elif local and remote:
                # Both exist — check for conflict
                if local.lifecycle != remote.lifecycle:
                    # Lifecycle conflict → Lamport resolves
                    winner = self._resolve_by_clock(local, remote)
                    merged.append(winner)
                elif self._content_hash(local) != self._content_hash(remote):
                    # Content conflict
                    code_confidence = 0.95  # We're good at this
                    if not self.dial.should_escalate(code_confidence):
                        winner = self._resolve_by_clock(local, remote)
                        merged.append(winner)
                    elif model_fn:
                        # Model decides which version to keep
                        keep_local = model_fn(
                            self._tile_data(local),
                            self._tile_data(remote),
                        )
                        merged.append(local if keep_local else remote)
                        used_model = True
                    else:
                        # Fallback: keep newer
                        merged.append(self._resolve_by_clock(local, remote))
                else:
                    merged.append(local)  # identical

        elapsed = (time.monotonic() - t0) * 1000
        return ChainResult(
            signal=Signal.model_needed if used_model else Signal.early_exit,
            tile=merged[0] if merged else MemoryTile("empty-sync", self.dial.name),
            output={
                "merged": merged,
                "conflicts": len(conflicts),
                "total": len(all_ids),
            },
            alpha_used=self.dial.effective_alpha(used_model),
            latency_ms=elapsed,
        )

    @staticmethod
    def _resolve_by_clock(local: MemoryTile, remote: MemoryTile) -> MemoryTile:
        """Resolve conflict using Lamport clock — higher value wins."""
        if local.clock.value >= remote.clock.value:
            return local
        return remote

    @staticmethod
    def _content_hash(tile: MemoryTile) -> str:
        """Hash tile payload for comparison."""
        return MemoryTile.content_hash(str(sorted(tile.payload.items())))

    @staticmethod
    def _tile_data(tile: MemoryTile) -> str:
        """Extract text data for model comparison."""
        if "chunk" in tile.payload:
            return tile.payload["chunk"].get("content", "")
        if "node" in tile.payload:
            return tile.payload["node"].get("summary", "")
        return str(tile.payload)
