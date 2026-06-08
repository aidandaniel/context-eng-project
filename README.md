# Context Engineering MCP

A local [Model Context Protocol](https://modelcontextprotocol.io) server that
returns **query-matched, token-budgeted context packs** instead of whole files.
The goal: cut the input tokens an agent spends gathering context while keeping
the information it actually needs.

On the bundled benchmark it delivers a **median 63.6% token reduction with 100%
anchor recall** (2,330 -> 762 median tokens). See `benchmarks/BASELINE.md`.

## How it works

```
query -> analyze_query -> intent + token budget
      -> get_context_bundle -> ranked, budgeted chunks
      -> expand_context (only if needed)
```

Instead of reading whole files, the server returns:

1. **Anchors** - explicitly mentioned files/symbols (always included).
2. **Symbol slices** - the relevant function/class, not the whole file.
3. **Import neighbors** - 1-hop dependencies of the anchors.
4. **Keyword snippets** - top grep matches with a few lines of context.

The token budget is a **ceiling, not a fill target**: low-value chunks are
dropped even when budget remains.

## Requirements

- Python 3.10+
- No `ripgrep` or `uv` required (a pure-Python grep fallback is used; `tiktoken`
  is used for token counting when installed, else a chars/4 heuristic).

## Setup

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

(omit `[dev]` for a runtime-only install; `[dev]` adds pytest and tiktoken.)

### Register with Cursor

`.cursor/mcp.json` launches the server from the venv (no fixed workspace required):

```json
{
  "mcpServers": {
    "context-eng": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "context_eng.server"]
    }
  }
}
```

For a **global** MCP config (same server for all projects), register once in
Cursor user settings with the same command. The agent should pass
`workspace_root` on each call (the open project path).

Optional env fallback in MCP config:

```json
"env": { "CONTEXT_ENG_WORKSPACE": "/path/to/default/project" }
```

Restart Cursor (or reload MCP servers) after changes. Copy
`.cursor/rules/context-eng-mcp.mdc` into other projects (or add as a user rule)
so the agent uses the tools consistently.

## Tools

| Tool | Purpose |
|------|---------|
| `analyze_query(query, workspace_root?)` | Detect intent, extract files/symbols, recommend a token budget. |
| `get_context_bundle(query, max_tokens?, intent?, workspace_root?)` | Return ranked, budgeted context chunks. |
| `expand_context(bundle_id, focus?, extra_tokens?)` | Progressive disclosure when a bundle is insufficient. |
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
refactor 10000, review 5000.

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
The pytest gate (`tests/test_benchmark.py`) enforces:

- median token reduction >= 30%
- median anchor recall >= 95%
- p90 latency < 3s

To see the "before tuning" effect, raise `min_chunk_score`/`max_optional_chunks`
limits or widen `grep_context_lines` and re-run; reduction will fall.

## Manual eval checklist

- [ ] Server appears under MCP settings with 4 tools.
- [ ] Run a debug query that names a file/symbol; confirm the bundle includes
      that file and is far smaller than the full file.
- [ ] Confirm `expand_context` adds chunks without duplicating the bundle.
- [ ] With the rule enabled, confirm the agent calls `get_context_bundle`
      instead of reading whole files for a multi-file task.

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
