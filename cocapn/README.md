# cocapn/ — PLATO Room Decomposition for OpenHuman

> **Every OpenHuman module, a PLATO room with a signal chain dial.**

OpenHuman has 60+ Rust domain modules. This directory maps each into PLATO rooms
where the signal chain architecture (per-stage α dial, tiles carrying context,
spectral conservation) can operate.

## Room Mapping (Initial)

| OpenHuman Module | PLATO Room | α | Why |
|-----------------|------------|---|-----|
| memory/ | memory-ingest | 0.2 | Code chunks, model for summarization |
| tree_summarizer/ | memory-tree | 0.3 | Code for fixed summarization, model for novel content |
| context/ | context-pipeline | 0.4 | Code for simple prompts, model for complex |
| routing/ | model-routing | 0.5 | Code for known providers, model for novel routing |
| inference/ | inference-gate | 0.3 | Code for cached/common, model for new queries |
| tokenjuice/ | token-compress | 0.1 | Mostly code (rules + regex), model for CJK/edge cases |
| agent/ | agent-dispatch | 0.6 | Code for known tasks, model for novel requests |
| integrations/ | integration-fetch | 0.2 | Code for API calls, model for error recovery |
| tools/ | tool-registry | 0.3 | Code for known tools, model for tool selection |

## Key Insight: OpenHuman's Memory Tree IS a PLATO Room Hierarchy

The Memory Tree (hierarchical summary trees stored in SQLite) maps directly
to PLATO's room structure. Each tree node can become a room with its own α dial.
The signal chain's early-exit mechanism means most memory queries resolve at
the code level (cached summaries) without needing to invoke a model.

## Integration Points

1. **Memory Tree → PLATO Tiles**: Each tree node becomes a tile
2. **Context Pipeline → Signal Chain**: Context stages become α-dial rooms
3. **TokenJuice → Spectral Conservation**: Token compression preserves information
4. **Routing → Model Gate**: Provider selection uses α dial logic
5. **Agent Dispatcher → Pipeline**: Task routing uses early-exit pattern

## See Also
- `cocapn/plato_rooms.py` — Room definitions (coming)
- `cocapn/analysis/` — Deep analysis of each OpenHuman module
- SuperInstance/spreader-tool — Signal chain implementation
- SuperInstance/plato-types — Tile lifecycle, Lamport clocks
