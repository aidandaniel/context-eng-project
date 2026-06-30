# Fix RF evaluation gates

The Random Forest budget classifier must pass three independent gates.
Read `ml/reports/rf_eval.md` and fix what failed.

## Failed gates
${FAILED_GATES}

## Gate definitions
1. **RF_CV_GATE** — 5-fold stratified CV on `ml/data/budget_labels.jsonl`
   - overall accuracy >= `min_cv_accuracy` in `ml/data/eval_targets.yaml`
   - per-bucket accuracy >= `min_cv_bucket_accuracy` for buckets with enough samples
2. **RF_ANCHOR_RECALL_GATE** — on `ml/data/budget_training_queries.yaml`
   - RF budget anchor recall >= `min_anchor_recall_rf`
   - RF recall >= intent-table recall (`min_anchor_recall_delta`)
3. **RF_AB_BENCHMARK_GATE** — intent vs RF on `benchmarks/queries.yaml`
   - RF median token reduction within `max_reduction_regression_pp` of intent table
   - RF is wired via `Config.budget_source="rf"` in `ContextEngine`

## Allowed fixes
- Improve training corpus / labels (`ml/data/budget_training_queries.yaml`)
- Retrain: `context-eng-ml-labels` then `context-eng-ml-train`
- Tune thresholds only in `ml/data/eval_targets.yaml` if justified
- Fix `src/context_eng/ml/engine_budget.py` or engine integration bugs

## Verify (each grep must exit 0 independently)
```
context-eng-ml-eval
./ml/scripts/validate_cv.sh
./ml/scripts/validate_anchor_recall.sh
./ml/scripts/validate_ab_benchmark.sh
```

Report which gates now PASS and paste the three gate lines from `ml/reports/rf_eval.md`.
