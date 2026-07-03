# Context Engineering MCP

A local [Model Context Protocol](https://modelcontextprotocol.io) server that
returns **query-matched, token-budgeted context packs** instead of whole files.
The goal: cut the input tokens an agent spends gathering context while keeping
the information it actually needs.

On the bundled benchmark (62 queries on the fixture repo), the RF-backed runtime
delivers a **median 63.5% token reduction** (~1,075 median MCP tokens; p90
latency ~119 ms). See `ml/reports/rf_eval.md` (gates) and
`benchmarks/results/latest.md` (per-query report from `context-eng-benchmark`).

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

Pipeline (no oracle labels at runtime):

1. **Retrieve** — grep matches first; optional local embeddings merge semantic hits.
2. **Discover anchors** — infer must-include files from query + repo (no user file list).
3. **Budget** — Random Forest picks a token ceiling; auto-fit bumps the bucket if anchors won't fit.
4. **Pack** — anchors, symbol slices, import neighbors, and top keyword snippets; adaptive cap on optional chunks.

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
pip install -e ".[dev,ml]"
```

Optional extras:

```bash
pip install -e ".[embeddings]"   # semantic retriever (sentence-transformers, off by default)
pip install -e ".[viz]"          # RF decision-tree visualization
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
max_optional_chunks_upper = 4 # adaptive cap ceiling (floor = 1)
max_inferred_anchor_files = 3
budget_source = "rf"          # rf | intent (intent is legacy)
enable_embedding_retriever = false
embedding_model_name = "all-MiniLM-L6-v2"
ignore_globs = [".git", "node_modules", "dist", "__pycache__"]

[context_eng.intent_budgets]
debug = [6000, 3000, 9000]    # [recommended, min, max] — RF features + UI only
implement = [8000, 4000, 12000]
```

**Budget resolution** (runtime): explicit `max_tokens` on the tool call → RF model
(`ml/models/budget_rf_v2.joblib`) → `default_max_tokens` snapped to the nearest
bucket. Missing model file uses the fallback bucket without crashing.

**Adaptive optional chunks**: when `max_optional_chunks` is unset, the cap scales
with budget size, discovered anchor count, and query length (median **2.0** on the
training fixture).

**Embeddings**: set `enable_embedding_retriever = true` and install `.[embeddings]`.
Grep stays primary; embedding hits are merged and deduped. Off by default — behavior
matches grep-only.

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
- p90 latency < 3s

Re-run `context-eng-benchmark` after config changes to refresh the checked-in report.

## ML evaluation

RF budget model training and CI gates:

```bash
context-eng-ml-labels    # sweep labels from inferred anchors
context-eng-ml-train     # train budget_rf_v2.joblib
context-eng-ml-eval      # write ml/reports/rf_eval.md + rf_eval_dashboard.png
```

`context-eng-ml-eval` also writes **`ml/reports/rf_eval_dashboard.png`** — label
distribution, CV confusion matrix, feature importance, and per-bucket precision/recall.

Current gates (`ml/reports/rf_eval.md`):

| Gate | Threshold | Latest |
|------|-----------|--------|
| TOKEN_REDUCTION | median >= 55% | **63.5%** |
| P90_LATENCY | < 3000 ms | **~119 ms** |
| ANCHOR_RETENTION | >= 90% | **100%** |
| RF_CV | >= 25% | **100%** |

Roadmap PRD and Ralph loop: `ml/prd/mcp_v2_roadmap.md`, `./ralph_mcp_v2.sh`.

## Manual eval checklist

- [ ] Run `.\scripts\install.ps1`, restart Cursor.
- [ ] MCP settings shows `context-eng` with `prepare_context` and `/context` prompt.
- [ ] Type `/context explain how refresh tokens work` — context chunks appear
      without manually calling tools.
- [ ] Confirm `expand_context` adds chunks without duplicating the bundle.

## Project layout

```
src/context_eng/      # server + engine (transport-independent core)
  server.py           # FastMCP tool wrappers
  engine.py           # orchestration: analyze / bundle / expand
  anchors/            # runtime anchor discovery + budget auto-fit
  intent/             # rule-based classifier + budget table (RF features)
  retrieval/          # grep, optional embeddings, composite merge
  ranking/            # weighted chunk ranker
  packing/            # adaptive optional-chunk cap
  budget/             # greedy token packing
  ml/                 # RF budget model, labels, eval gates
  tokens/             # tiktoken / chars-4 estimator
  logging/            # append-only JSONL event log (ML-ready)
benchmarks/           # fixture repo, queries, baseline/MCP runners, report
ml/                   # training data, models, PRDs, Ralph scripts
tests/                # unit tests + benchmark gate
```

## Roadmap

- Reverse-dependency edges to raise supporting-context recall.
- Persistent embedding index (v1 embeds on demand per query).
- Task-success / human eval beyond token-reduction gates.
