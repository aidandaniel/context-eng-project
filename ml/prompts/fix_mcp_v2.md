# Fix MCP v2 roadmap (phase ${MCP_V2_PHASE:-auto})

You are implementing the Context-Eng MCP v2 roadmap. Read:
- `ml/prd/mcp_v2_roadmap.prd.json` (active phase requirements)
- `ml/reports/rf_eval.md` (cumulative gate lines)
- `ml/reports/mcp_v2_status.md` (phase markers)

## Failed checks
${FAILED_GATES}

## Active phase
${ACTIVE_PHASE_NAME}

Phase marker required in `ml/reports/mcp_v2_status.md`:
`${ACTIVE_PHASE_MARKER}`

## Cumulative gates (always blocking)
1. **TOKEN_REDUCTION_GATE** — RF runtime, median reduction **>= 55%**
2. **P90_LATENCY_GATE** — p90 < 3000 ms
3. **ANCHOR_RETENTION_GATE** — runtime anchors in bundle >= 90%
4. **RF_CV_GATE** — CV overall accuracy >= 25%

## Phase-specific guidance

### Phase 1 — RF default at runtime
- Set `Config.budget_source = "rf"` default in `config.py`
- `engine_budget.py`: explicit max_tokens → RF → `default_max_tokens` bucket snap on missing model
- Remove intent-table runtime budget branch; keep intent for RF features only
- Decouple `budget_model.predict()` from `fixed_budget.recommended` bump
- Update tests in `tests/test_rf_eval.py`, `tests/test_benchmark.py`
- Append `PHASE1_RF_DEFAULT: PASS` to `mcp_v2_status.md` when done

### Phase 2 — Budget auto-fit
- Add `src/context_eng/anchors/fit.py` (`estimate_must_include_tokens`, `ensure_budget_fits_anchors`)
- Wire in `engine.get_context_bundle` after budget resolve, before candidates
- Refactor `_build_candidates` to use `discover_anchor_paths`
- Use `hard_ceiling_factor=1.5`; bump to next `BUDGET_BUCKETS` entry only when needed
- Add `tests/test_anchor_fit.py`; optional audit `BUDGET_AUTOFIT_GATE: PASS` in rf_eval.md
- Append `PHASE2_BUDGET_AUTOFIT: PASS` when done

### Phase 3 — Inferred-anchor labels
- Replace oracle sweep with `sweep_one_query_inferred` at production Config (no `_TRAINING_CONFIG_OVERRIDES`)
- `label_source: inferred_sweep`; `expected_anchors` → audit `oracle_anchor_recall` only
- Regenerate: `context-eng-ml-labels` → `context-eng-ml-train`
- Append `PHASE3_INFERRED_LABELS: PASS` when done

### Phase 4 — Embedding retriever (optional extra)
- `pyproject.toml` optional `embeddings = ["sentence-transformers>=3.0"]`
- `embedding_retriever.py`; `enable_embedding_retriever=False` by default
- Composite retriever in engine; regression test with embeddings off
- Append `PHASE4_EMBEDDING_RETRIEVER: PASS` and `REGRESSION_NO_EMBEDDINGS_GATE: PASS` when done

### Phase 5 — Adaptive optional chunk cap
- `adaptive_max_optional_chunks()` — base 2, +1 for budget>=8000, anchors>=4, query_tokens>=40; cap at `max_optional_chunks_upper=4`
- Replace fixed `max_optional_chunks=2` default
- Report `median_optional_chunks_used` in eval
- Append `PHASE5_ADAPTIVE_OPTIONAL_CAP: PASS` when done

## Allowed fixes
- Code changes scoped to active phase files (see PRD `phases[].files`)
- Retrain pipeline when labels/model change
- Tune packing within phase policy; do **not** lower 55% threshold without user approval

## Forbidden
- Using `expected_anchors` at runtime
- Intent budget table as runtime ceiling
- Lowering gate thresholds in `eval_targets.yaml` without approval

## Verify
```
context-eng-ml-labels && context-eng-ml-train && context-eng-ml-eval || true
./ml/scripts/validate_mcp_v2_gates.sh
pytest -m "not benchmark"
python -m mypy src/context_eng benchmarks
```

Report cumulative gate lines from `rf_eval.md` and the phase marker you added.
