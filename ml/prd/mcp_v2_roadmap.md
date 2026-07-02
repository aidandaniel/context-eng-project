# Context-Eng MCP v2 Roadmap

Machine-readable spec: [`mcp_v2_roadmap.prd.json`](mcp_v2_roadmap.prd.json)  
Extends: [`rf_only_self_contained.prd.json`](rf_only_self_contained.prd.json) (55% token reduction, RF eval harness)

## North star

On any single query + open repo, **without oracle labels**:

1. Discover anchors from query + repo
2. RF picks a token ceiling; auto-fit ensures anchors fit
3. Pack context with adaptive optional chunks
4. Grep + optional local embeddings improve recall
5. Train RF on **inferred** anchors, not `expected_anchors`

**Implementation order:** Phase 1 → 2 → 3 → 5 → 4 (embeddings after stable anchor/budget behavior).

## Baseline (post–Ralph RF self-contained)

| Metric | Value |
|--------|-------|
| TOKEN_REDUCTION (RF) | 63.6% |
| ANCHOR_RETENTION | 100% |
| Production `budget_source` | `intent` (Phase 1 fixes) |
| Label generation | oracle `expected_anchors` |
| Retriever | `GrepRetriever` only |
| `max_optional_chunks` | 2 (fixed) |

---

## Phase 1 — RF default at runtime

**Problem:** Eval uses RF; `/context` still uses the intent budget table.

**Work:** Default `budget_source="rf"`. Resolution order: explicit `max_tokens` → RF model → `default_max_tokens` bucket snap. Remove intent-table runtime branch. Decouple `budget_model.predict()` from intent-table floor bump.

**Done when:** `PHASE1_RF_DEFAULT: PASS` in `ml/reports/mcp_v2_status.md` and cumulative gates pass.

---

## Phase 2 — Budget auto-fit for anchors

**Problem:** RF bucket may be too low for must-include anchors.

**Work:** New `anchors/fit.py`; `ensure_budget_fits_anchors()` after `resolve_budget_limit`; refactor `_build_candidates` to use `discover_anchor_paths`.

**Done when:** `PHASE2_BUDGET_AUTOFIT: PASS`, ANCHOR_RETENTION ≥ 90% with `max_inferred_anchor_files=6`, TOKEN_REDUCTION ≥ 55%.

---

## Phase 3 — Train labels on inferred anchors

**Problem:** Labels use oracle anchors and training overrides.

**Work:** `sweep_one_query_inferred()` at production config; `label_source: inferred_sweep`; remove `_TRAINING_CONFIG_OVERRIDES`; regenerate labels + retrain.

**Done when:** `PHASE3_INFERRED_LABELS: PASS`, RF_CV ≥ 25%, TOKEN_REDUCTION ≥ 55%.

---

## Phase 4 — Local embedding retriever (optional, off by default)

**Problem:** Grep misses semantic matches.

**Work:** `pip install -e ".[embeddings]"`; `EmbeddingRetriever`; composite engine retriever; `enable_embedding_retriever=False` by default.

**Done when:** `PHASE4_EMBEDDING_RETRIEVER: PASS`, `REGRESSION_NO_EMBEDDINGS_GATE: PASS`, embeddings-off behavior unchanged.

**Non-goals:** Vector DB, cloud APIs, persistent index (v1).

---

## Phase 5 — Adaptive optional chunk cap

**Problem:** Fixed `max_optional_chunks=2` trades recall on broad queries for benchmark reduction.

**Work:** `adaptive_max_optional_chunks()` — base 2, boosts for large budget / many anchors / long query; upper bound 4.

**Done when:** `PHASE5_ADAPTIVE_OPTIONAL_CAP: PASS`, TOKEN_REDUCTION ≥ 55%, `median_optional_chunks_used` in report.

---

## Cumulative CI gates (blocking)

| Gate | Threshold |
|------|-----------|
| TOKEN_REDUCTION_GATE | median ≥ **55%** (RF runtime) |
| P90_LATENCY_GATE | < 3000 ms |
| ANCHOR_RETENTION_GATE | ≥ **90%** |
| RF_CV_GATE | ≥ **25%** |
| REGRESSION_NO_EMBEDDINGS_GATE | Phase 4+ only |

Phase markers (grep in `ml/reports/mcp_v2_status.md`):

```
PHASE1_RF_DEFAULT: PASS
PHASE2_BUDGET_AUTOFIT: PASS
PHASE3_INFERRED_LABELS: PASS
PHASE4_EMBEDDING_RETRIEVER: PASS
PHASE5_ADAPTIVE_OPTIONAL_CAP: PASS
```

---

## Ralph loop

```bash
# Full roadmap: each phase must grep PASS before advancing (exit 0 at end)
./ralph_mcp_v2.sh

# Grep-verify only (no agent, no train/eval)
VERIFY_ONLY=1 ./ralph_mcp_v2.sh

# Single phase
MCP_V2_PHASE=2 ./ralph_mcp_v2.sh

# Resume from phase 3
START_PHASE=3 ./ralph_mcp_v2.sh

# Manual grep check
./ml/scripts/validate_mcp_v2_gates.sh --phase 1
./ml/scripts/validate_mcp_v2_gates.sh --all
```

- Reads `mcp_v2_roadmap.prd.json`
- Runs labels → train → eval each iteration
- Greps cumulative gates in `ml/reports/rf_eval.md`
- Greps active phase marker in `ml/reports/mcp_v2_status.md`
- Agent fix prompt: `ml/prompts/fix_mcp_v2.md`

---

## Out of scope

- Cloud embedding APIs
- Persistent vector index / LSP
- Removing intent classifier (RF feature + UI)
- Raising 55% gate without approval
- Task-success human eval (Phase 6)
