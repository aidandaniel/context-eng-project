# Fix RF evaluation gates

The Random Forest budget classifier must pass four independent self-contained gates.
Read `ml/reports/rf_eval.md` and fix what failed.

## Failed gates
${FAILED_GATES}

## Gate definitions
1. **TOKEN_REDUCTION_GATE** — RF runtime on `benchmarks/queries.yaml`
   - median token reduction >= `min_median_reduction_pct` in `ml/data/eval_targets.yaml`
2. **P90_LATENCY_GATE** — RF runtime on `benchmarks/queries.yaml`
   - p90 latency < `max_p90_latency_ms` in `ml/data/eval_targets.yaml`
3. **ANCHOR_RETENTION_GATE** — RF runtime on `ml/data/budget_training_queries.yaml`
   - all runtime-discovered anchor paths present in bundle >= `min_anchor_retention`
4. **RF_CV_GATE** — 5-fold stratified CV on `ml/data/budget_labels.jsonl`
   - overall accuracy >= `min_cv_accuracy` in `ml/data/eval_targets.yaml`

## Allowed fixes
- Improve training corpus / labels (`ml/data/budget_training_queries.yaml`)
- Retrain: `context-eng-ml-labels` then `context-eng-ml-train`
- Tune thresholds only in `ml/data/eval_targets.yaml` if justified
- Fix `src/context_eng/ml/engine_budget.py` or engine integration bugs

## Verify
```
context-eng-ml-eval
./ml/scripts/validate_rf_self_contained_gates.sh
```

Report which gates now PASS and paste the four gate lines from `ml/reports/rf_eval.md`.
