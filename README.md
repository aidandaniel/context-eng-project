# Context Engineering MCP

A local [Model Context Protocol](https://modelcontextprotocol.io) server that
returns **query-matched, token-budgeted context packs** instead of whole files.
The goal: cut the input tokens an agent spends gathering context while keeping
the information it actually needs.

On the bundled benchmark (62 queries on the fixture repo) it delivers a
**median 49.5% token reduction** (2,660 -> 1,559 median tokens; p90 latency
~97 ms). See `benchmarks/results/latest.md`.

## Quick start

```powershell
# one-time install (registers MCP globally + copies /context command)
.\scripts\install.ps1
```

Restart Cursor, then in any project:

```
/context how does auth middleware validate tokens?
```

That single slash command analyzes your query, picks a token budget, fetches the
right code slices, and injects them into the chat. No manual tool juggling.

## How it works

```
/context <query>  ->  prepare_context  ->  budgeted chunks (ready to use)
                                      ->  expand_context (only if needed)
```

Instead of reading whole files, the server returns:

1. **Anchors** - explicitly mentioned files/symbols (always included).
2. **Symbol slices** - the relevant function/class, not the whole file.
3. **Import neighbors** - 1-hop dependencies of the anchors.
4. **Keyword snippets** - top grep matches with a few lines of context.

The token budget is a **ceiling, not a fill target**: low-value chunks are
dropped even when budget remains.

## Requirements

- Python 3.11+
- No `ripgrep` or `uv` required (a pure-Python grep fallback is used; `tiktoken`
  is used for token counting when installed, else a chars/4 heuristic).

## Setup

### Option A — one command (recommended)

```powershell
.\scripts\install.ps1
```

This creates a venv, installs the package, registers the MCP server in
`~/.cursor/mcp.json`, and copies the `/context` slash command to
`~/.cursor/commands/`. Restart Cursor when it finishes.

### Option B — manual

```powershell
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Add to `~/.cursor/mcp.json` (global, works for all projects):

```json
{
  "mcpServers": {
    "context-eng": {
      "command": "C:/path/to/context-eng-project/.venv/Scripts/python.exe",
      "args": ["-m", "context_eng.server"]
    }
  }
}
```

Copy `.cursor/commands/context.md` to `~/.cursor/commands/` (or use the MCP
prompt `/context` that the server registers automatically).

Optional: copy `.cursor/rules/context-eng-mcp.mdc` into other projects so the
agent prefers `prepare_context` even without the slash command.

## Tools

| Tool / command | Purpose |
|----------------|---------|
| **`/context <query>`** (MCP prompt) | One-step UX: analyze + bundle, inject formatted context into chat. |
| **`prepare_context(query, ...)`** | Same as `/context` but as an MCP tool (for agents). |
| `expand_context(bundle_id, focus?, extra_tokens?)` | Add more context only when the initial bundle is insufficient. |
| `analyze_query` / `get_context_bundle` | Lower-level tools (used internally; prefer `prepare_context`). |
| `estimate_tokens(text? \| bundle_id?)` | Token count for text or a built bundle. |

### Workspace resolution (multi-project)

| Priority | Source |
|----------|--------|
| 1 | `workspace_root` argument on the tool call |
| 2 | `CONTEXT_ENG_WORKSPACE` env var in MCP config |
| 3 | Process cwd (typically set to the open project) |

Tool responses include `workspace_root` so you can confirm which repo was indexed.

## Configuration

Optional `context-eng.toml` at the workspace root overrides defaults:

```toml
[context_eng]
default_max_tokens = 8000
grep_context_lines = 8
max_grep_candidates = 50
min_chunk_score = 0.15        # drop optional chunks below this score
max_optional_chunks = 4       # cap non-anchor chunks
ignore_globs = [".git", "node_modules", "dist", "__pycache__"]

[context_eng.intent_budgets]
debug = [6000, 3000, 9000]    # [recommended, min, max]
implement = [8000, 4000, 12000]
```

Intent -> budget defaults: debug 6000, implement 8000, explain 4000,
refactor 10000, review 5000. Runtime token ceilings use the trained RF model
(`budget_source = "rf"` by default); the intent table feeds classifier features only.

## Tests

```bash
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pytest                              # everything (incl. benchmark gate)
pytest -m "not benchmark"           # fast unit tests only
pytest -m benchmark                 # benchmark gate only
```

## Benchmark (before vs after)

Quantifies token impact by running the same queries two ways: a **baseline**
that reads top grep-matched files in full, and the **MCP** budgeted bundle.

```bash
python -m benchmarks.compare
# or, after install: context-eng-benchmark
```

Writes `benchmarks/results/latest.{json,md}` and prints an aggregate summary.
Latest checked-in results: **49.5%** median reduction, **1,559** median MCP
tokens (baseline **2,660**), **97 ms** p90 latency across **62** queries.

The pytest gate (`tests/test_benchmark.py`) enforces:

- median token reduction >= 30%
- p90 latency < 3s

To see the "before tuning" effect, raise `min_chunk_score`/`max_optional_chunks`
limits or widen `grep_context_lines` and re-run; reduction will fall.

## Manual eval checklist

- [ ] Run `.\scripts\install.ps1`, restart Cursor.
- [ ] MCP settings shows `context-eng` with `prepare_context` and `/context` prompt.
- [ ] Type `/context explain how refresh tokens work` — context chunks appear
      without manually calling tools.
- [ ] Confirm `expand_context` adds chunks without duplicating the bundle.

## Project layout

```
src/context_eng/      # the server + engine (transport-independent core)
  server.py           # FastMCP tool wrappers
  engine.py           # orchestration: analyze / bundle / expand
  intent/             # rule-based classifier + budget table
  retrieval/          # grep retriever, symbol slicing, import graph
  ranking/            # weighted chunk ranker
  budget/             # greedy token packing
  tokens/             # tiktoken / chars-4 estimator
  logging/            # append-only JSONL event log (ML-ready)
benchmarks/           # fixture repo, queries, baseline/MCP runners, report
tests/                # unit tests + benchmark gate
```

## Roadmap (v2)

- `EmbeddingRetriever` behind the existing `Retriever` protocol.
- Reverse-dependency edges to raise supporting-context recall.
- Train a budget model from `.context-eng/events.jsonl` (the logged features are
  already ML-ready); blend predicted budgets with the fixed table.
