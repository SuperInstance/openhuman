"""
PLATO rooms for OpenHuman context pipeline and routing.

Four rooms, each a signal-chain stage (tiles in, α dial, early exit):

  ContextBuildRoom   (α=0.3) — assemble context from tiles/prompt sections
  ContextCompressRoom (α=0.1) — compress history to fit token budget
  ModelRouteRoom      (α=0.4) — route request to model provider
  InferenceGateRoom   (α=0.3) — manage inference call (cache or dispatch)

Each room receives tiles, processes them through a signal chain, and
emits output tiles. The α parameter controls how aggressively the room
delegates to code vs. model: low α → code-heavy (deterministic), high α
→ model-heavy (learned/adaptive).

Design mirrors the Rust codebase:
  - context/pipeline.rs  → ContextBuildRoom + ContextCompressRoom
  - context/prompt.rs    → ContextBuildRoom (prompt assembly)
  - context/summarizer.rs → ContextCompressRoom (LLM summarization)
  - routing/policy.rs    → ModelRouteRoom (task classification + routing)
  - routing/provider.rs  → ModelRouteRoom + InferenceGateRoom (dispatch)
  - inference/types.rs   → shared types across all rooms
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ─── Shared types ─────────────────────────────────────────────────────────────

class TileKind(str, Enum):
    """Tile kinds flowing through the signal chain."""
    PROMPT_SECTION = "prompt_section"
    HISTORY_ENTRY = "history_entry"
    TOOL_RESULT = "tool_result"
    USAGE_INFO = "usage_info"
    ROUTING_HINT = "routing_hint"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    SUMMARY = "summary"
    SIGNAL = "signal"


@dataclass
class Tile:
    """A single tile flowing through a PLATO room.

    Tiles carry typed payloads and metadata for tracing. Each tile is
    content-addressed (SHA-256 of payload bytes) for dedup and caching.
    """
    kind: TileKind
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @property
    def content_hash(self) -> str:
        raw = f"{self.kind.value}:{self.payload}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class RoomOutput:
    """Output from a room's signal chain.

    `tiles`: the emitted output tiles.
    `early_exit`: if True, downstream rooms should skip processing.
    `stats`: room-local metrics for observability.
    """
    tiles: list[Tile] = field(default_factory=list)
    early_exit: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


class Signal(str, Enum):
    """Control signals that flow between rooms."""
    OK = "ok"
    NOOP = "noop"
    COMPACTION_NEEDED = "compaction_needed"
    CONTEXT_EXHAUSTED = "context_exhausted"
    AUTOCOMPACTION_REQUESTED = "autocompaction_requested"
    AUTOCOMPACTION_DISABLED = "autocompaction_disabled"
    ROUTE_LOCAL = "route_local"
    ROUTE_REMOTE = "route_remote"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    FALLBACK_TRIGGERED = "fallback_triggered"


class TaskCategory(str, Enum):
    """Task complexity tier for routing — mirrors routing/policy.rs."""
    LIGHTWEIGHT = "lightweight"
    MEDIUM = "medium"
    HEAVY = "heavy"


class RoutingTarget(str, Enum):
    """Where to send a request — mirrors routing/policy.rs."""
    LOCAL = "local"
    REMOTE = "remote"


# ─── Token budgeting (mirrors tool_result_budget.rs) ──────────────────────────

DEFAULT_TOOL_RESULT_BUDGET_BYTES: int = 16 * 1024  # 16 KiB
DEFAULT_CONTEXT_WINDOW: int = 128_000
SOFT_THRESHOLD_PCT: float = 0.80
HARD_THRESHOLD_PCT: float = 0.95


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def apply_tool_result_budget(
    content: str, budget_bytes: int = DEFAULT_TOOL_RESULT_BUDGET_BYTES
) -> tuple[str, dict[str, Any]]:
    """Truncate oversized tool results before they enter history.

    Mirrors Rust `apply_tool_result_budget`. Returns (content, stats).
    """
    original = len(content)
    if budget_bytes == 0 or original <= budget_bytes:
        return content, {"original_bytes": original, "final_bytes": original, "truncated": False}

    trailer_reserved = 256
    head_capacity = max(1, budget_bytes - trailer_reserved)
    cut = min(head_capacity, len(content))
    dropped = original - cut

    truncated = content[:cut] + (
        f"\n\n[… {dropped} bytes truncated by tool_result_budget "
        "— re-run with a narrower query to see the rest …]"
    )
    return truncated, {
        "original_bytes": original,
        "final_bytes": len(truncated),
        "truncated": True,
    }


# ─── Task classification (mirrors routing/policy.rs classify()) ───────────────

LIGHTWEIGHT_HINTS = {"reaction", "classify", "format", "sentiment", "lightweight"}
MEDIUM_HINTS = {"summarize", "medium", "tool_lite"}


def classify_task(model: str) -> TaskCategory:
    """Classify a model string (possibly hint:*) into a task category.

    Mirrors Rust `policy::classify`.
    """
    if model.startswith("hint:"):
        hint = model[5:]
        if hint in LIGHTWEIGHT_HINTS:
            return TaskCategory.LIGHTWEIGHT
        if hint in MEDIUM_HINTS:
            return TaskCategory.MEDIUM
    return TaskCategory.HEAVY


def decide_route(
    category: TaskCategory,
    local_available: bool,
    privacy_required: bool = False,
    latency_low: bool = False,
    cost_high: bool = False,
) -> tuple[RoutingTarget, Optional[RoutingTarget]]:
    """Decide where to route. Returns (primary, fallback).

    Mirrors Rust `policy::decide`.
    """
    if privacy_required:
        return RoutingTarget.LOCAL, None

    if category == TaskCategory.HEAVY:
        return RoutingTarget.REMOTE, None

    local_bias = int(latency_low) + int(cost_high)
    use_local = local_available and (
        category == TaskCategory.LIGHTWEIGHT
        or (category == TaskCategory.MEDIUM and local_bias > 0)
    )

    if use_local:
        return RoutingTarget.LOCAL, RoutingTarget.REMOTE
    return RoutingTarget.REMOTE, None


# ─── Session memory state (mirrors context/session_memory.rs) ─────────────────

DEFAULT_MIN_TOKEN_GROWTH: int = 4_000
DEFAULT_MIN_TOOL_CALLS: int = 8
DEFAULT_MIN_TURNS_BETWEEN: int = 4


@dataclass
class SessionMemoryConfig:
    min_token_growth: int = DEFAULT_MIN_TOKEN_GROWTH
    min_tool_calls: int = DEFAULT_MIN_TOOL_CALLS
    min_turns_between: int = DEFAULT_MIN_TURNS_BETWEEN


@dataclass
class SessionMemoryState:
    """Per-session extraction state — mirrors Rust SessionMemoryState."""
    total_tokens: int = 0
    tokens_at_last_extract: int = 0
    turn_at_last_extract: int = 0
    total_tool_calls: int = 0
    tool_calls_at_last_extract: int = 0
    current_turn: int = 0
    extraction_in_progress: bool = False

    def tick_turn(self) -> None:
        self.current_turn += 1

    def record_usage(self, total: int) -> None:
        if total > self.total_tokens:
            self.total_tokens = total

    def record_tool_calls(self, n: int) -> None:
        self.total_tool_calls += n

    def should_extract(self, cfg: SessionMemoryConfig) -> bool:
        if self.extraction_in_progress:
            return False
        return (
            (self.total_tokens - self.tokens_at_last_extract) >= cfg.min_token_growth
            and (self.total_tool_calls - self.tool_calls_at_last_extract) >= cfg.min_tool_calls
            and (self.current_turn - self.turn_at_last_extract) >= cfg.min_turns_between
        )

    def mark_started(self) -> None:
        self.extraction_in_progress = True

    def mark_complete(self) -> None:
        self.extraction_in_progress = False
        self.tokens_at_last_extract = self.total_tokens
        self.tool_calls_at_last_extract = self.total_tool_calls
        self.turn_at_last_extract = self.current_turn

    def mark_failed(self) -> None:
        self.extraction_in_progress = False


# ─── Context guard (mirrors context/guard.rs) ────────────────────────────────

@dataclass
class ContextGuard:
    """Track context utilization and circuit breaker state."""
    input_tokens: int = 0
    output_tokens: int = 0
    context_window: int = DEFAULT_CONTEXT_WINDOW
    consecutive_failures: int = 0
    max_failures: int = 3

    def update_usage(self, usage: dict[str, int]) -> None:
        self.input_tokens = usage.get("input_tokens", self.input_tokens)
        self.output_tokens = usage.get("output_tokens", self.output_tokens)
        self.context_window = usage.get("context_window", self.context_window)

    @property
    def utilization(self) -> float:
        if self.context_window == 0:
            return 0.0
        return (self.input_tokens + self.output_tokens) / self.context_window

    @property
    def utilization_pct(self) -> int:
        return int(self.utilization * 100)

    def check(self) -> Signal:
        """Check context state. Mirrors ContextCheckResult."""
        if self.consecutive_failures >= self.max_failures and self.utilization > HARD_THRESHOLD_PCT:
            return Signal.CONTEXT_EXHAUSTED
        if self.utilization > SOFT_THRESHOLD_PCT:
            return Signal.COMPACTION_NEEDED
        return Signal.OK

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1


# ─── Room base class ──────────────────────────────────────────────────────────

class Room:
    """Base PLATO room with signal chain pattern.

    α controls code-vs-model delegation:
      low α  → code path (deterministic, fast)
      high α → model path (adaptive, learned)

    Each room processes tiles through:
      1. ingest()   — receive and validate input tiles
      2. process()  — apply room logic (code or model path)
      3. emit()     — produce output tiles
    """

    def __init__(self, name: str, alpha: float):
        self.name = name
        self.alpha = alpha  # 0..1, higher = more model delegation
        self._stats: dict[str, Any] = {"calls": 0, "code_path": 0, "model_path": 0}

    def _use_code_path(self) -> bool:
        """Decide code vs. model path based on α."""
        import random
        return random.random() > self.alpha

    def ingest(self, tiles: list[Tile]) -> list[Tile]:
        """Validate and filter input tiles."""
        return tiles

    def process(self, tiles: list[Tile]) -> list[Tile]:
        """Process tiles through the room's signal chain."""
        raise NotImplementedError

    def emit(self, processed: list[Tile]) -> RoomOutput:
        """Package processed tiles into room output."""
        return RoomOutput(tiles=processed)

    def run(self, tiles: list[Tile]) -> RoomOutput:
        """Execute the full signal chain: ingest → process → emit."""
        self._stats["calls"] += 1
        ingested = self.ingest(tiles)
        processed = self.process(ingested)
        return self.emit(processed)


# ─── ContextBuildRoom ─────────────────────────────────────────────────────────

class ContextBuildRoom(Room):
    """Build context from tiles — mirrors context/pipeline.rs + prompt.rs.

    α=0.3: mostly code (prompt assembly is deterministic).
    Model path: adaptive prompt section ordering for complex multi-agent sessions.

    Signal chain:
      1. Collect prompt sections from tiles
      2. Apply tool result budget inline
      3. Assemble system prompt + history
      4. Early exit if context fits budget
    """

    def __init__(
        self,
        alpha: float = 0.3,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        tool_budget_bytes: int = DEFAULT_TOOL_RESULT_BUDGET_BYTES,
    ):
        super().__init__("ContextBuildRoom", alpha)
        self.context_window = context_window
        self.tool_budget_bytes = tool_budget_bytes
        self.guard = ContextGuard(context_window=context_window)

    def process(self, tiles: list[Tile]) -> list[Tile]:
        output: list[Tile] = []
        prompt_parts: list[str] = []
        history_entries: list[str] = []

        for tile in tiles:
            if tile.kind == TileKind.PROMPT_SECTION:
                prompt_parts.append(str(tile.payload))
                output.append(tile)

            elif tile.kind == TileKind.TOOL_RESULT:
                content, stats = apply_tool_result_budget(
                    str(tile.payload), self.tool_budget_bytes
                )
                output.append(Tile(
                    kind=TileKind.TOOL_RESULT,
                    payload=content,
                    metadata={"budget_stats": stats},
                ))

            elif tile.kind == TileKind.HISTORY_ENTRY:
                history_entries.append(str(tile.payload))
                output.append(tile)

            elif tile.kind == TileKind.USAGE_INFO:
                self.guard.update_usage(tile.payload if isinstance(tile.payload, dict) else {})
                output.append(tile)

            else:
                output.append(tile)

        # Assemble system prompt tile
        if prompt_parts:
            system_prompt = "\n\n".join(prompt_parts)
            output.insert(0, Tile(
                kind=TileKind.PROMPT_SECTION,
                payload=system_prompt,
                metadata={"sections": len(prompt_parts), "assembled": True},
            ))

        # Emit context check signal
        signal = self.guard.check()
        output.append(Tile(kind=TileKind.SIGNAL, payload=signal.value))

        self._stats["code_path"] += 1
        return output

    def emit(self, processed: list[Tile]) -> RoomOutput:
        # Early exit if guard is OK
        signals = [t for t in processed if t.kind == TileKind.SIGNAL]
        early_exit = any(s.payload == Signal.OK.value for s in signals)
        return RoomOutput(
            tiles=processed,
            early_exit=early_exit,
            stats={
                "utilization_pct": self.guard.utilization_pct,
                "signal": signals[-1].payload if signals else Signal.OK.value,
            },
        )


# ─── ContextCompressRoom ──────────────────────────────────────────────────────

class ContextCompressRoom(Room):
    """Compress context to fit budget — mirrors microcompact + summarizer.

    α=0.1: almost entirely code (token counting + truncation).
    Model path: LLM summarization for autocompaction.

    Signal chain:
      1. Check context guard signal
      2. If COMPACTION_NEEDED → run microcompact (code path)
      3. If still over budget → request autocompaction (model path)
      4. If CONTEXT_EXHAUSTED → early exit with error signal
    """

    def __init__(
        self,
        alpha: float = 0.1,
        keep_recent: int = 5,
        max_failures: int = 3,
    ):
        super().__init__("ContextCompressRoom", alpha)
        self.keep_recent = keep_recent
        self.max_failures = max_failures
        self.consecutive_failures = 0

    def ingest(self, tiles: list[Tile]) -> list[Tile]:
        # Only process if there's a compaction signal
        return tiles

    def _microcompact(self, tiles: list[Tile]) -> tuple[list[Tile], dict[str, Any]]:
        """Code path: clear old tool results, keep recent ones.

        Mirrors Rust microcompact — replaces old ToolResult payloads
        with a placeholder while preserving tile count (API invariant).
        """
        cleared = 0
        bytes_freed = 0
        tool_results = [(i, t) for i, t in enumerate(tiles)
                        if t.kind == TileKind.TOOL_RESULT]

        total_tool = len(tool_results)
        preserve = min(self.keep_recent, total_tool)

        for idx, (i, tile) in enumerate(tool_results):
            if idx < total_tool - preserve:
                original_len = len(str(tile.payload))
                tiles[i] = Tile(
                    kind=TileKind.TOOL_RESULT,
                    payload="[cleared by microcompact]",
                    metadata={"original_bytes": original_len, "cleared": True},
                )
                cleared += 1
                bytes_freed += original_len

        stats = {"envelopes_cleared": cleared, "bytes_freed": bytes_freed}
        return tiles, stats

    def _autocompact(self, tiles: list[Tile]) -> tuple[list[Tile], dict[str, Any]]:
        """Model path: summarize older history entries.

        For the PLATO room, this is a stub that marks tiles for summarization.
        In production, this would dispatch to an LLM summarizer.
        """
        history = [t for t in tiles if t.kind == TileKind.HISTORY_ENTRY]
        if len(history) <= self.keep_recent:
            return tiles, {"messages_removed": 0, "tokens_freed": 0}

        head_count = len(history) - self.keep_recent
        summary = f"[auto-compacted] Summary of {head_count} earlier messages"
        bytes_freed = sum(len(str(t.payload)) for t in history[:head_count])

        # Replace history tiles with summary
        new_tiles: list[Tile] = []
        history_idx = 0
        for tile in tiles:
            if tile.kind == TileKind.HISTORY_ENTRY:
                if history_idx < head_count:
                    new_tiles.append(Tile(
                        kind=TileKind.SUMMARY,
                        payload=summary,
                        metadata={"compacted_count": head_count},
                    ))
                    # Only emit one summary tile
                    history_idx = head_count
                else:
                    new_tiles.append(tile)
                history_idx += 1
            else:
                new_tiles.append(tile)

        stats = {"messages_removed": head_count, "tokens_freed": bytes_freed // 4}
        return new_tiles, stats

    def process(self, tiles: list[Tile]) -> list[Tile]:
        # Find the signal tile
        signal_tile = next(
            (t for t in tiles if t.kind == TileKind.SIGNAL), None
        )
        signal = signal_tile.payload if signal_tile else Signal.OK.value

        if signal == Signal.OK.value:
            self._stats["code_path"] += 1
            return tiles

        if signal == Signal.CONTEXT_EXHAUSTED.value:
            self._stats["code_path"] += 1
            return tiles + [Tile(
                kind=TileKind.SIGNAL,
                payload=Signal.CONTEXT_EXHAUSTED.value,
            )]

        # COMPACTION_NEEDED — try microcompact first (code path)
        tiles, micro_stats = self._microcompact(tiles)
        self._stats["code_path"] += 1

        if micro_stats["envelopes_cleared"] > 0:
            self.consecutive_failures = 0
            tiles.append(Tile(
                kind=TileKind.SIGNAL,
                payload=Signal.OK.value,
                metadata={"microcompacted": micro_stats},
            ))
            return tiles

        # Microcompact didn't help — autocompact (model path)
        if self.alpha > 0 and self.consecutive_failures < self.max_failures:
            self._stats["model_path"] += 1
            tiles, auto_stats = self._autocompact(tiles)
            if auto_stats["messages_removed"] > 0:
                self.consecutive_failures = 0
                tiles.append(Tile(
                    kind=TileKind.SIGNAL,
                    payload=Signal.OK.value,
                    metadata={"autocompacted": auto_stats},
                ))
                return tiles

        # Still over budget
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            tiles.append(Tile(
                kind=TileKind.SIGNAL,
                payload=Signal.CONTEXT_EXHAUSTED.value,
            ))
        else:
            tiles.append(Tile(
                kind=TileKind.SIGNAL,
                payload=Signal.AUTOCOMPACTION_REQUESTED.value,
            ))
        return tiles

    def emit(self, processed: list[Tile]) -> RoomOutput:
        signals = [t for t in processed if t.kind == TileKind.SIGNAL]
        last_signal = signals[-1].payload if signals else Signal.OK.value
        early_exit = last_signal in (Signal.OK.value, Signal.CONTEXT_EXHAUSTED.value)
        return RoomOutput(
            tiles=processed,
            early_exit=early_exit,
            stats={"last_signal": last_signal, "consecutive_failures": self.consecutive_failures},
        )


# ─── ModelRouteRoom ───────────────────────────────────────────────────────────

class ModelRouteRoom(Room):
    """Route to model provider — mirrors routing/policy.rs + factory.rs.

    α=0.4: balanced — code for known providers/hints, model for novel routing.

    Signal chain:
      1. Classify task from model hint
      2. Check local health
      3. Apply routing policy (decide_route)
      4. Emit routing target tile
    """

    def __init__(
        self,
        alpha: float = 0.4,
        local_available: bool = False,
        local_model: str = "local-model",
        remote_model: str = "remote-model",
    ):
        super().__init__("ModelRouteRoom", alpha)
        self.local_available = local_available
        self.local_model = local_model
        self.remote_model = remote_model
        self._route_cache: dict[str, tuple[RoutingTarget, Optional[RoutingTarget]]] = {}

    def process(self, tiles: list[Tile]) -> list[Tile]:
        output = list(tiles)

        for tile in tiles:
            if tile.kind != TileKind.MODEL_REQUEST:
                continue

            model = str(tile.payload)
            metadata = tile.metadata.copy()

            # Code path: classify + decide
            category = classify_task(model)
            hints = metadata.get("routing_hints", {})
            privacy = hints.get("privacy_required", False)
            latency_low = hints.get("latency_budget") == "low"
            cost_high = hints.get("cost_sensitivity") == "high"

            primary, fallback = decide_route(
                category,
                self.local_available,
                privacy_required=privacy,
                latency_low=latency_low,
                cost_high=cost_high,
            )

            # Cache the decision
            cache_key = f"{model}:{category.value}:{privacy}:{latency_low}:{cost_high}"
            self._route_cache[cache_key] = (primary, fallback)

            resolved_model = (
                self.local_model if primary == RoutingTarget.LOCAL
                else self.remote_model
            )

            route_tile = Tile(
                kind=TileKind.ROUTING_HINT,
                payload={
                    "primary": primary.value,
                    "fallback": fallback.value if fallback else None,
                    "category": category.value,
                    "resolved_model": resolved_model,
                    "original_hint": model,
                },
                metadata=metadata,
            )
            output.append(route_tile)
            self._stats["code_path"] += 1

        return output

    def emit(self, processed: list[Tile]) -> RoomOutput:
        routes = [t for t in processed if t.kind == TileKind.ROUTING_HINT]
        return RoomOutput(
            tiles=processed,
            early_exit=False,
            stats={"routes_decided": len(routes)},
        )


# ─── InferenceGateRoom ────────────────────────────────────────────────────────

class InferenceGateRoom(Room):
    """Manage inference calls — mirrors inference/provider.rs dispatch.

    α=0.3: code for cached/known queries, model for new complex queries.

    Signal chain:
      1. Check response cache
      2. If cache hit → emit cached response, early exit
      3. If cache miss → emit dispatch signal with resolved route
    """

    def __init__(
        self,
        alpha: float = 0.3,
        max_cache_size: int = 256,
    ):
        super().__init__("InferenceGateRoom", alpha)
        self._cache: dict[str, Tile] = {}
        self.max_cache_size = max_cache_size
        self._hits = 0
        self._misses = 0

    def _cache_key(self, model: str, message: str) -> str:
        return hashlib.sha256(f"{model}:{message}".encode()).hexdigest()[:16]

    def process(self, tiles: list[Tile]) -> list[Tile]:
        output: list[Tile] = []

        # Collect route info and request
        route_info: dict[str, Any] | None = None
        request: Tile | None = None

        for tile in tiles:
            if tile.kind == TileKind.ROUTING_HINT and isinstance(tile.payload, dict):
                route_info = tile.payload
            elif tile.kind == TileKind.MODEL_REQUEST:
                request = tile
            output.append(tile)

        if not request:
            return output

        model = route_info.get("resolved_model", "") if route_info else ""
        message = str(request.payload)
        key = self._cache_key(model, message)

        # Check cache (code path)
        if key in self._cache:
            self._hits += 1
            self._stats["code_path"] += 1
            cached = self._cache[key]
            output.append(Tile(
                kind=TileKind.MODEL_RESPONSE,
                payload=cached.payload,
                metadata={"cache_hit": True, "model": model},
            ))
            output.append(Tile(
                kind=TileKind.SIGNAL,
                payload=Signal.CACHE_HIT.value,
            ))
            return output

        # Cache miss — dispatch needed
        self._misses += 1
        self._stats["code_path"] += 1

        # Add to cache (placeholder — in production, the actual response
        # would be cached after the provider returns)
        if len(self._cache) < self.max_cache_size:
            self._cache[key] = Tile(
                kind=TileKind.MODEL_RESPONSE,
                payload=f"[dispatched to {model}]",
            )

        output.append(Tile(
            kind=TileKind.SIGNAL,
            payload=Signal.CACHE_MISS.value,
            metadata={"model": model, "cache_key": key},
        ))
        return output

    def emit(self, processed: list[Tile]) -> RoomOutput:
        signals = [t for t in processed if t.kind == TileKind.SIGNAL]
        last_signal = signals[-1].payload if signals else ""
        early_exit = last_signal == Signal.CACHE_HIT.value
        return RoomOutput(
            tiles=processed,
            early_exit=early_exit,
            stats={
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "cache_size": len(self._cache),
                "last_signal": last_signal,
            },
        )

    def inject_response(self, model: str, message: str, response: str) -> None:
        """Pre-populate the cache with a known response."""
        key = self._cache_key(model, message)
        self._cache[key] = Tile(
            kind=TileKind.MODEL_RESPONSE,
            payload=response,
        )


# ─── Pipeline orchestrator ────────────────────────────────────────────────────

class ContextPipeline:
    """Orchestrate the four rooms in sequence — mirrors ContextPipeline in Rust.

    Tiles flow through:
      ContextBuildRoom → ContextCompressRoom → ModelRouteRoom → InferenceGateRoom

    Early exit at any stage short-circuits downstream rooms.
    """

    def __init__(
        self,
        build_room: ContextBuildRoom | None = None,
        compress_room: ContextCompressRoom | None = None,
        route_room: ModelRouteRoom | None = None,
        inference_room: InferenceGateRoom | None = None,
    ):
        self.build = build_room or ContextBuildRoom()
        self.compress = compress_room or ContextCompressRoom()
        self.route = route_room or ModelRouteRoom()
        self.inference = inference_room or InferenceGateRoom()

    def run(self, tiles: list[Tile]) -> dict[str, RoomOutput]:
        """Run tiles through all rooms. Returns per-room outputs."""
        results: dict[str, RoomOutput] = {}

        # Stage 1: Build context
        out1 = self.build.run(tiles)
        results["build"] = out1
        if out1.early_exit:
            return results

        # Stage 2: Compress
        out2 = self.compress.run(out1.tiles)
        results["compress"] = out2
        if out2.early_exit:
            return results

        # Stage 3: Route
        out3 = self.route.run(out2.tiles)
        results["route"] = out3

        # Stage 4: Inference gate
        out4 = self.inference.run(out3.tiles)
        results["inference"] = out4

        return results
