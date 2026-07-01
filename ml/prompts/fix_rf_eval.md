# Fix RF evaluation gates

The Random Forest budget classifier must pass four independent self-contained gates.
Read `ml/reports/rf_eval.md` and fix what failed.

## Failed gates
${FAILED_GATES}

## Gate definitions (PRD baseline)
1. **TOKEN_REDUCTION_GATE** — RF runtime on `benchmarks/queries.yaml`
   - median token reduction **>= 55%** (`min_median_reduction_pct` in `ml/data/eval_targets.yaml`)
   - grep must find `TOKEN_REDUCTION_GATE: PASS` with `threshold_pct=55`
2. **P90_LATENCY_GATE** — RF runtime on `benchmarks/queries.yaml`
   - p90 latency < `max_p90_latency_ms` (3000 ms)
3. **ANCHOR_RETENTION_GATE** — RF runtime on `ml/data/budget_training_queries.yaml`
   - runtime-discovered anchor paths present in bundle >= 90%
4. **RF_CV_GATE** — stratified CV on `ml/data/budget_labels.jsonl`
   - overall accuracy >= 25%

## Allowed fixes
- Improve training corpus / labels (`ml/data/budget_training_queries.yaml`); retrain with `context-eng-ml-labels` then `context-eng-ml-train`
- Tune packing (`max_optional_chunks`, `min_chunk_score`) in `context_eng.config` / `context-eng.toml`
- Fix `src/context_eng/ml/engine_budget.py`, `anchors/discovery.py`, or engine integration bugs
- Do **not** lower `min_median_reduction_pct` below 55 without explicit user approval

## Verify
```
context-eng-ml-eval
./ml/scripts/validate_rf_self_contained_gates.sh
```

Report which gates now PASS and paste the four gate lines from `ml/reports/rf_eval.md`.
