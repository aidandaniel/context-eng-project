# RF evaluation report

RELEVANT_FILE_RECALL_GATE: PASS recall=0.917 threshold_recall=0.70
TASK_SUCCESS_GATE: PASS task_success=1.000 threshold_task_success=0.80
TOKEN_REDUCTION_GATE: PASS median_reduction=63.0 threshold_pct=55
P90_LATENCY_GATE: PASS p90_latency_ms=215.3
INFERRED_ANCHOR_RETENTION: retention=1.000
RF_CV_GATE: PASS overall=0.316 min_bucket=0.000
LABEL_BUCKET_SPREAD: PASS buckets=9 min_buckets=3
RETRIEVAL_P90_GATE: PASS p90_ms=152.5 threshold=3000
BUDGET_AUTOFIT_GATE: PASS bumped=0/266
oracle_anchor_recall: median=0.250 quality_sweep_rows=266/266
median_optional_chunks_used=2.0
REGRESSION_NO_EMBEDDINGS_GATE: PASS mismatches=0/266
embedding_eval_recall: skipped model_available=False queries=4

## Task quality (task_eval_queries.yaml)
- relevant file recall: 0.917
- task rubric success: 1.000

## RF benchmark (queries.yaml)
- median token reduction: 63.0%
- p90 latency: 215.3 ms
- median MCP tokens: 1076

## Budget auto-fit (budget_training_queries.yaml)
- queries bumped: 0/266

## Adaptive optional chunks (budget_training_queries.yaml)
- median_optional_chunks_used=2.0

## Embedding regression (embeddings off)
- REGRESSION_NO_EMBEDDINGS_GATE: PASS mismatches=0/266

## Embedding recall audit (embedding_eval_queries.yaml)
- embedding_eval_recall: skipped model_available=False queries=4

## Inferred anchor retention (budget_training_queries.yaml)
- retention rate: 1.000

## 5-fold CV (budget_labels.jsonl)
- overall accuracy: 0.316
- per-bucket accuracy:
  - 2000: 0.298 (n=47)
  - 3000: 0.422 (n=45)
  - 4000: 0.294 (n=34)
  - 5000: 0.560 (n=50)
  - 6000: 0.000 (n=22)
  - 8000: 0.045 (n=22)
  - 10000: 0.333 (n=12)
  - 12000: 0.273 (n=22)
  - 15000: 0.167 (n=12)

## Dashboard
- visualization: `C:/Users/decke/context-eng-project/ml/reports/rf_eval_dashboard.png`
