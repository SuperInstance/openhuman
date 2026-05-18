"""Tests for PLATO context pipeline rooms — at least 15 tests."""

import pytest

from cocapn.context_rooms import (
    # Types
    Tile, TileKind, Signal, TaskCategory, RoutingTarget,
    RoomOutput, SessionMemoryConfig, SessionMemoryState, ContextGuard,
    # Functions
    estimate_tokens, apply_tool_result_budget, classify_task, decide_route,
    # Rooms
    ContextBuildRoom, ContextCompressRoom, ModelRouteRoom, InferenceGateRoom,
    ContextPipeline,
    # Constants
    DEFAULT_CONTEXT_WINDOW, DEFAULT_TOOL_RESULT_BUDGET_BYTES,
    SOFT_THRESHOLD_PCT, HARD_THRESHOLD_PCT,
)


# ─── Token estimation ─────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world") == len("hello world") // 4

    def test_empty_string(self):
        assert estimate_tokens("") == 1

    def test_long_text(self):
        text = "a" * 4000
        assert estimate_tokens(text) == 1000


# ─── Tool result budget ───────────────────────────────────────────────────────

class TestToolResultBudget:
    def test_small_content_passes_through(self):
        content = "hello"
        result, stats = apply_tool_result_budget(content, 1024)
        assert result == content
        assert not stats["truncated"]

    def test_oversized_content_truncated(self):
        content = "x" * 10000
        result, stats = apply_tool_result_budget(content, 1024)
        assert stats["truncated"]
        assert len(result) < len(content)
        assert "truncated by tool_result_budget" in result

    def test_zero_budget_noop(self):
        content = "keep me"
        result, stats = apply_tool_result_budget(content, 0)
        assert result == content
        assert not stats["truncated"]

    def test_exact_budget_unchanged(self):
        content = "x" * 100
        result, stats = apply_tool_result_budget(content, 100)
        assert not stats["truncated"]


# ─── Task classification ──────────────────────────────────────────────────────

class TestClassifyTask:
    def test_lightweight_hints(self):
        for hint in ["hint:reaction", "hint:classify", "hint:format",
                      "hint:sentiment", "hint:lightweight"]:
            assert classify_task(hint) == TaskCategory.LIGHTWEIGHT, f"{hint}"

    def test_medium_hints(self):
        for hint in ["hint:summarize", "hint:medium", "hint:tool_lite"]:
            assert classify_task(hint) == TaskCategory.MEDIUM, f"{hint}"

    def test_heavy_hints(self):
        for hint in ["hint:reasoning", "hint:chat", "hint:agentic", "hint:coding"]:
            assert classify_task(hint) == TaskCategory.HEAVY, f"{hint}"

    def test_exact_model_is_heavy(self):
        assert classify_task("gemma3:4b-it-qat") == TaskCategory.HEAVY
        assert classify_task("") == TaskCategory.HEAVY


# ─── Routing decision ─────────────────────────────────────────────────────────

class TestDecideRoute:
    def test_lightweight_local_healthy(self):
        primary, fallback = decide_route(TaskCategory.LIGHTWEIGHT, True)
        assert primary == RoutingTarget.LOCAL
        assert fallback == RoutingTarget.REMOTE

    def test_lightweight_local_unavailable(self):
        primary, fallback = decide_route(TaskCategory.LIGHTWEIGHT, False)
        assert primary == RoutingTarget.REMOTE
        assert fallback is None

    def test_heavy_always_remote(self):
        primary, fallback = decide_route(TaskCategory.HEAVY, True)
        assert primary == RoutingTarget.REMOTE
        assert fallback is None

    def test_privacy_forces_local(self):
        for cat in [TaskCategory.LIGHTWEIGHT, TaskCategory.MEDIUM, TaskCategory.HEAVY]:
            primary, fallback = decide_route(cat, True, privacy_required=True)
            assert primary == RoutingTarget.LOCAL
            assert fallback is None

    def test_medium_with_latency_bias(self):
        primary, fallback = decide_route(
            TaskCategory.MEDIUM, True, latency_low=True
        )
        assert primary == RoutingTarget.LOCAL

    def test_medium_without_bias_goes_remote(self):
        primary, _ = decide_route(TaskCategory.MEDIUM, True)
        assert primary == RoutingTarget.REMOTE


# ─── Session memory state ─────────────────────────────────────────────────────

class TestSessionMemoryState:
    def test_default_no_extract(self):
        state = SessionMemoryState()
        cfg = SessionMemoryConfig()
        assert not state.should_extract(cfg)

    def test_all_thresholds_crossed(self):
        cfg = SessionMemoryConfig()
        state = SessionMemoryState(
            total_tokens=cfg.min_token_growth + 1,
            total_tool_calls=cfg.min_tool_calls + 1,
            current_turn=cfg.min_turns_between + 1,
        )
        assert state.should_extract(cfg)

    def test_in_progress_suppresses(self):
        cfg = SessionMemoryConfig()
        state = SessionMemoryState(
            total_tokens=10000, total_tool_calls=20, current_turn=10,
        )
        assert state.should_extract(cfg)
        state.mark_started()
        assert not state.should_extract(cfg)

    def test_mark_complete_resets(self):
        cfg = SessionMemoryConfig()
        state = SessionMemoryState(total_tokens=10000, total_tool_calls=20, current_turn=10)
        state.mark_started()
        state.mark_complete()
        assert not state.should_extract(cfg)

    def test_mark_failed_leaves_deltas(self):
        cfg = SessionMemoryConfig()
        state = SessionMemoryState(
            total_tokens=cfg.min_token_growth + 1,
            total_tool_calls=cfg.min_tool_calls + 1,
            current_turn=cfg.min_turns_between + 1,
        )
        state.mark_started()
        state.mark_failed()
        assert state.should_extract(cfg)

    def test_record_usage_monotonic(self):
        state = SessionMemoryState()
        state.record_usage(5000)
        state.record_usage(3000)
        assert state.total_tokens == 5000

    def test_tick_turn(self):
        state = SessionMemoryState()
        state.tick_turn()
        state.tick_turn()
        assert state.current_turn == 2


# ─── Context guard ────────────────────────────────────────────────────────────

class TestContextGuard:
    def test_ok_when_under_threshold(self):
        guard = ContextGuard(input_tokens=1000, context_window=100000)
        assert guard.check() == Signal.OK

    def test_compaction_needed_at_soft_threshold(self):
        guard = ContextGuard(input_tokens=85000, context_window=100000)
        assert guard.check() == Signal.COMPACTION_NEEDED

    def test_exhausted_with_circuit_breaker(self):
        guard = ContextGuard(
            input_tokens=96000, context_window=100000,
            consecutive_failures=3,
        )
        assert guard.check() == Signal.CONTEXT_EXHAUSTED

    def test_record_success_resets_failures(self):
        guard = ContextGuard(consecutive_failures=2)
        guard.record_success()
        assert guard.consecutive_failures == 0


# ─── ContextBuildRoom ─────────────────────────────────────────────────────────

class TestContextBuildRoom:
    def test_assembles_prompt_sections(self):
        room = ContextBuildRoom()
        tiles = [
            Tile(kind=TileKind.PROMPT_SECTION, payload="You are helpful."),
            Tile(kind=TileKind.PROMPT_SECTION, payload="Time: noon."),
        ]
        output = room.run(tiles)
        assembled = [t for t in output.tiles
                     if t.kind == TileKind.PROMPT_SECTION
                     and t.metadata.get("assembled")]
        assert len(assembled) == 1
        assert "You are helpful." in assembled[0].payload
        assert "Time: noon." in assembled[0].payload

    def test_truncates_oversized_tool_results(self):
        room = ContextBuildRoom()
        big_result = "x" * 20000
        tiles = [Tile(kind=TileKind.TOOL_RESULT, payload=big_result)]
        output = room.run(tiles)
        result_tiles = [t for t in output.tiles if t.kind == TileKind.TOOL_RESULT]
        assert len(result_tiles) == 1
        assert len(result_tiles[0].payload) < len(big_result)

    def test_early_exit_when_ok(self):
        room = ContextBuildRoom()
        tiles = [Tile(kind=TileKind.PROMPT_SECTION, payload="hi")]
        output = room.run(tiles)
        assert output.early_exit is True


# ─── ContextCompressRoom ──────────────────────────────────────────────────────

class TestContextCompressRoom:
    def test_microcompact_clears_old_tool_results(self):
        room = ContextCompressRoom(keep_recent=2)
        tiles = [
            Tile(kind=TileKind.TOOL_RESULT, payload="old1"),
            Tile(kind=TileKind.TOOL_RESULT, payload="old2"),
            Tile(kind=TileKind.TOOL_RESULT, payload="recent1"),
            Tile(kind=TileKind.TOOL_RESULT, payload="recent2"),
            Tile(kind=TileKind.SIGNAL, payload=Signal.COMPACTION_NEEDED.value),
        ]
        output = room.run(tiles)
        tool_tiles = [t for t in output.tiles if t.kind == TileKind.TOOL_RESULT]
        cleared = [t for t in tool_tiles if t.metadata.get("cleared")]
        assert len(cleared) == 2

    def test_passes_through_on_ok_signal(self):
        room = ContextCompressRoom()
        tiles = [
            Tile(kind=TileKind.HISTORY_ENTRY, payload="hello"),
            Tile(kind=TileKind.SIGNAL, payload=Signal.OK.value),
        ]
        output = room.run(tiles)
        assert output.early_exit is True

    def test_autocompact_replaces_old_history(self):
        room = ContextCompressRoom(keep_recent=2)
        tiles = [
            Tile(kind=TileKind.HISTORY_ENTRY, payload=f"msg{i}")
            for i in range(10)
        ] + [Tile(kind=TileKind.SIGNAL, payload=Signal.COMPACTION_NEEDED.value)]
        output = room.run(tiles)
        summaries = [t for t in output.tiles if t.kind == TileKind.SUMMARY]
        assert len(summaries) >= 1


# ─── ModelRouteRoom ───────────────────────────────────────────────────────────

class TestModelRouteRoom:
    def test_classifies_and_routes(self):
        room = ModelRouteRoom(local_available=True, local_model="local", remote_model="remote")
        tiles = [Tile(kind=TileKind.MODEL_REQUEST, payload="hint:reaction")]
        output = room.run(tiles)
        routes = [t for t in output.tiles if t.kind == TileKind.ROUTING_HINT]
        assert len(routes) == 1
        assert routes[0].payload["primary"] == RoutingTarget.LOCAL.value

    def test_heavy_goes_remote(self):
        room = ModelRouteRoom(local_available=True)
        tiles = [Tile(kind=TileKind.MODEL_REQUEST, payload="hint:reasoning")]
        output = room.run(tiles)
        routes = [t for t in output.tiles if t.kind == TileKind.ROUTING_HINT]
        assert routes[0].payload["primary"] == RoutingTarget.REMOTE.value

    def test_privacy_forces_local(self):
        room = ModelRouteRoom(local_available=False)
        tiles = [Tile(
            kind=TileKind.MODEL_REQUEST,
            payload="hint:reasoning",
            metadata={"routing_hints": {"privacy_required": True}},
        )]
        output = room.run(tiles)
        routes = [t for t in output.tiles if t.kind == TileKind.ROUTING_HINT]
        assert routes[0].payload["primary"] == RoutingTarget.LOCAL.value


# ─── InferenceGateRoom ────────────────────────────────────────────────────────

class TestInferenceGateRoom:
    def test_cache_miss_on_first_call(self):
        room = InferenceGateRoom()
        tiles = [
            Tile(kind=TileKind.MODEL_REQUEST, payload="hello"),
            Tile(kind=TileKind.ROUTING_HINT, payload={
                "resolved_model": "model-a",
                "primary": "remote",
            }),
        ]
        output = room.run(tiles)
        assert output.stats["last_signal"] == Signal.CACHE_MISS.value

    def test_cache_hit_on_repeat(self):
        room = InferenceGateRoom()
        room.inject_response("model-a", "hello", "cached response")

        tiles = [
            Tile(kind=TileKind.MODEL_REQUEST, payload="hello"),
            Tile(kind=TileKind.ROUTING_HINT, payload={
                "resolved_model": "model-a",
                "primary": "remote",
            }),
        ]
        output = room.run(tiles)
        assert output.early_exit is True
        responses = [t for t in output.tiles if t.kind == TileKind.MODEL_RESPONSE]
        assert any(t.payload == "cached response" for t in responses)

    def test_cache_size_limit(self):
        room = InferenceGateRoom(max_cache_size=2)
        for i in range(5):
            tiles = [
                Tile(kind=TileKind.MODEL_REQUEST, payload=f"msg{i}"),
                Tile(kind=TileKind.ROUTING_HINT, payload={"resolved_model": "m"}),
            ]
            room.run(tiles)
        assert room._stats["code_path"] > 0  # room was called
        assert len(room._cache) <= 2


# ─── Full pipeline ────────────────────────────────────────────────────────────

class TestContextPipeline:
    def test_full_pipeline_runs_all_rooms(self):
        # Force no early exit by setting high utilization on the build room
        build = ContextBuildRoom()
        build.guard = ContextGuard(
            input_tokens=85000, context_window=100000,
        )
        pipeline = ContextPipeline(build_room=build)
        tiles = [
            Tile(kind=TileKind.PROMPT_SECTION, payload="System prompt."),
            Tile(kind=TileKind.MODEL_REQUEST, payload="hint:reaction"),
        ]
        results = pipeline.run(tiles)
        assert "build" in results
        assert "compress" in results

    def test_pipeline_early_exit_on_ok(self):
        pipeline = ContextPipeline()
        tiles = [Tile(kind=TileKind.PROMPT_SECTION, payload="Simple prompt.")]
        results = pipeline.run(tiles)
        # Build room returns OK for low utilization → early exit
        assert results["build"].early_exit is True
        # Downstream rooms should not run
        assert "compress" not in results


# ─── Tile content hashing ─────────────────────────────────────────────────────

class TestTile:
    def test_content_hash_deterministic(self):
        t1 = Tile(kind=TileKind.PROMPT_SECTION, payload="hello")
        t2 = Tile(kind=TileKind.PROMPT_SECTION, payload="hello")
        assert t1.content_hash == t2.content_hash

    def test_different_payload_different_hash(self):
        t1 = Tile(kind=TileKind.PROMPT_SECTION, payload="hello")
        t2 = Tile(kind=TileKind.PROMPT_SECTION, payload="world")
        assert t1.content_hash != t2.content_hash
