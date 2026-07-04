# Context Engineering MCP

A local [Model Context Protocol](https://modelcontextprotocol.io) server that
returns **query-matched, token-budgeted context packs** instead of whole files.

## Goals

On any single query against an open repo — **without oracle labels or a user-supplied
file list**:

1. **Retrieve** relevant code (grep-first; optional local embeddings).
2. **Discover anchors** — infer must-include files from the query + repo.
3. **Budget** — Random Forest picks a token ceiling; auto-fit bumps the bucket if
   anchors won't fit.
4. **Pack** — symbol slices, import neighbors, and keyword snippets under that ceiling.
5. **Prove quality** — CI gates enforce token reduction *and* relevant-file recall /
   task rubrics, not just path hits.

The north star: **cut agent input tokens while keeping the context an agent actually
needs to succeed on the task.**

## Current results

Latest ML gate report: `ml/reports/rf_eval.md` (RF runtime on the fixture repo).

| Gate | Threshold | Latest |
|------|-----------|--------|
| TOKEN_REDUCTION | median ≥ 55% | **63.0%** |
| P90_LATENCY | < 3000 ms | **~215 ms** |
| RELEVANT_FILE_RECALL | ≥ 70% | **91.7%** |
| TASK_SUCCESS | ≥ 80% | **100%** |
| RF_CV | ≥ 25% | **31.6%** |
| RETRIEVAL_P90 | < 3000 ms | **~153 ms** |
| LABEL_BUCKET_SPREAD | ≥ 3 buckets | **9 buckets** |

**Shipped roadmaps:** MCP v2 (RF default, budget auto-fit, inferred labels, adaptive
optional chunks, optional embeddings) and MCP v3 (manifest + ripgrep retrieval,
quality eval, recall-aware labels) — all phase markers **PASS** in
`ml/reports/mcp_v2_status.md` and `ml/reports/mcp_v3_status.md`.

Per-query before/after report: `benchmarks/results/latest.md` (regenerate with
`context-eng-benchmark`).

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

Pipeline:

1. **Index** — cached `.context-eng/manifest.json` avoids full-tree scans each query.
2. **Retrieve** — ripgrep when available (Python scan fallback); optional embeddings merge semantic hits.
3. **Discover anchors** — infer must-include files from query + repo.
4. **Budget** — RF model (`ml/models/budget_rf_v2.joblib`) picks a token ceiling;
   auto-fit raises the bucket if anchors won't fit.
5. **Pack** — anchors, symbol slices, import neighbors, and top keyword snippets;
   adaptive cap on optional chunks (median **2.0** on the training fixture).

The token budget is a **ceiling, not a fill target**: low-value chunks are dropped
even when budget remains.

## Requirements

- Python 3.11+
- **ripgrep** recommended (used when on `PATH`; pure-Python grep fallback otherwise)
- `tiktoken` optional for exact token counts (chars/4 heuristic otherwise)

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
pip install -e ".[viz]"          # RF decision-tree visualization (dtreeviz)
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
manifest_auto_build = true    # cache .context-eng/manifest.json
enable_embedding_retriever = false
embedding_model_name = "all-MiniLM-L6-v2"
ignore_globs = [".git", "node_modules", "dist", "__pycache__"]

[context_eng.intent_budgets]
debug = [6000, 3000, 9000]    # [recommended, min, max] — RF features + UI only
implement = [8000, 4000, 12000]
```

**Budget resolution** (runtime): explicit `max_tokens` on the tool call → RF model
→ `default_max_tokens` snapped to the nearest bucket. Missing model file uses the
fallback bucket without crashing.

**Adaptive optional chunks**: when `max_optional_chunks` is unset, the cap scales
with budget size, discovered anchor count, and query length.

**Embeddings**: set `enable_embedding_retriever = true` and install `.[embeddings]`.
Grep stays primary; embedding hits are merged and deduped. Off by default — behavior
matches grep-only (`REGRESSION_NO_EMBEDDINGS_GATE` in eval).

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

The pytest gate (`tests/test_benchmark.py`) enforces a looser MVP bar:

- median token reduction >= 30%
- p90 latency < 3s

Blocking ML gates (55% reduction, quality recall, etc.) live in `context-eng-ml-eval`.

## ML evaluation

RF budget model training, quality gates, and dashboards:

```bash
context-eng-ml-labels    # sweep labels from inferred anchors (quality-aware)
context-eng-ml-train     # train budget_rf_v2.joblib
context-eng-ml-eval      # write ml/reports/rf_eval.md + rf_eval_dashboard.png
```

`context-eng-ml-eval` writes:

- **`ml/reports/rf_eval.md`** — grep-verified gate lines for CI / Ralph loops
- **`ml/reports/rf_eval_dashboard.png`** — label distribution, CV confusion matrix,
  feature importance, per-bucket precision/recall (requires `.[ml]` / matplotlib)

### Visualizations

Use the **project venv** — global installs often lack optional deps.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[ml]"      # eval dashboard (matplotlib)
pip install -e ".[viz]"     # decision-tree SVG (dtreeviz + Graphviz dot)

context-eng-ml-eval                              # rf_eval_dashboard.png
context-eng-ml-viz-tree                          # ml/reports/budget_rf_tree.svg
# or: .\.venv\Scripts\python.exe -m context_eng.ml.visualize_forest
```

Tree visualization needs the native [Graphviz](https://graphviz.org/download/)
`dot` executable on `PATH` (Windows: often `C:\Program Files\Graphviz\bin`).

### Ralph loops (agent-driven PRD verification)

```bash
./ralph_mcp_v2.sh                    # v2: RF default, auto-fit, inferred labels, …
./ralph_mcp_v3.sh                    # v3: quality gates, manifest retrieval, …
VERIFY_ONLY=1 ./ralph_mcp_v3.sh      # grep-verify only (no agent)
./ml/scripts/validate_mcp_v3_gates.sh --all
```

PRDs: `ml/prd/mcp_v2_roadmap.md`, `ml/prd/mcp_v3_quality_retrieval.md`.

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
  index/              # cached workspace manifest
  eval/               # relevant-file recall + task rubric checks
  intent/             # rule-based classifier + budget table (RF features)
  retrieval/          # grep/ripgrep, optional embeddings, composite merge
  ranking/            # weighted chunk ranker
  packing/            # adaptive optional-chunk cap
  budget/             # greedy token packing
  ml/                 # RF budget model, labels, eval gates, dashboards
  tokens/             # tiktoken / chars-4 estimator
  logging/            # append-only JSONL event log (ML-ready)
benchmarks/           # fixture repo, queries, baseline/MCP runners, report
ml/                   # training data, models, PRDs, Ralph scripts, reports
tests/                # unit tests + benchmark gate
```

## Roadmap

**Done (v2 + v3):** RF-default runtime, anchor budget auto-fit, inferred-anchor
training labels, adaptive optional chunks, optional local embeddings (off by default),
manifest-backed ripgrep retrieval, blocking quality gates (relevant-file recall +
task rubrics).

**Next:**

- Persistent embedding index (v1 embeds on demand per query).
- Reverse-dependency edges to raise supporting-context recall.
- Broader task-success / human eval beyond the fixture corpus.
- Embedding recall audit when `.[embeddings]` is installed (`embedding_eval_recall`
  in `ml/reports/rf_eval.md`).
