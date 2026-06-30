# RF evaluation report

RF_CV_GATE: PASS overall=0.407 min_bucket=0.190
RF_ANCHOR_RECALL_GATE: PASS rf=0.922 intent=0.603 delta=0.319
RF_AB_BENCHMARK_GATE: PASS rf_reduction=49.5 intent_reduction=49.5 rf_mcp=1559 intent_mcp=1559 regression_pp=0.0

## 5-fold CV (budget_labels.jsonl)
- overall accuracy: 0.407
- per-bucket accuracy (buckets with enough samples):
  - 2000: 0.447 (n=76)
  - 6000: 0.191 (n=21)
  - 8000: 0.585 (n=65)
  - 10000: 0.217 (n=23)

## Anchor recall (budget_training_queries.yaml)
- RF recall: 0.922
- Intent recall: 0.603
- Delta (RF - intent): 0.319

## A/B token benchmark (benchmarks/queries.yaml)
- Intent median reduction: 49.5%
- RF median reduction: 49.5%
- Regression (intent - RF): 0.0 pp
- Intent median MCP tokens: 1559
- RF median MCP tokens: 1559
