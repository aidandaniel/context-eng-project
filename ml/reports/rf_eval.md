# RF evaluation report

TOKEN_REDUCTION_GATE: PASS median_reduction=63.5 threshold_pct=55
P90_LATENCY_GATE: PASS p90_latency_ms=118.9
ANCHOR_RETENTION_GATE: PASS retention=1.000
RF_CV_GATE: PASS overall=1.000 min_bucket=1.000
BUDGET_AUTOFIT_GATE: PASS bumped=0/266
oracle_anchor_recall: median=0.250 inferred_sweep_rows=246/266
median_optional_chunks_used=2.0
REGRESSION_NO_EMBEDDINGS_GATE: PASS mismatches=0/266
embedding_eval_recall: skipped model_available=False queries=4

## RF benchmark (queries.yaml)
- median token reduction: 63.5%
- p90 latency: 118.9 ms
- median MCP tokens: 1075

## Budget auto-fit (budget_training_queries.yaml)
- queries bumped: 0/266

## Adaptive optional chunks (budget_training_queries.yaml)
- median_optional_chunks_used=2.0

## Embedding regression (embeddings off)
- REGRESSION_NO_EMBEDDINGS_GATE: PASS mismatches=0/266

## Embedding recall audit (embedding_eval_queries.yaml)
- embedding_eval_recall: skipped model_available=False queries=4

## Anchor retention (budget_training_queries.yaml)
- retention rate: 1.000

## 5-fold CV (budget_labels.jsonl)
- overall accuracy: 1.000
- per-bucket accuracy:
  - 2000: 1.000 (n=266)
