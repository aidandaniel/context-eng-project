# RF evaluation report

RF_CV_GATE: FAIL overall=0.235 min_bucket=0.091 weak_buckets=5000=0.091,6000=0.091
RF_ANCHOR_RECALL_GATE: FAIL rf=0.328 intent=0.186 delta=0.142
RF_AB_BENCHMARK_GATE: PASS rf_reduction=49.5 intent_reduction=49.5 rf_mcp=1559 intent_mcp=1559 regression_pp=0.0

## 5-fold CV (budget_labels.jsonl)
- overall accuracy: 0.235
- per-bucket accuracy (buckets with enough samples):
  - 2000: 0.350 (n=20)
  - 3000: 0.300 (n=20)
  - 4000: 0.250 (n=20)
  - 5000: 0.091 (n=22)
  - 6000: 0.091 (n=22)
  - 8000: 0.160 (n=25)
  - 10000: 0.208 (n=24)
  - 12000: 0.500 (n=22)
  - 15000: 0.207 (n=29)

## Anchor recall (budget_training_queries.yaml)
- RF recall: 0.328
- Intent recall: 0.186
- Delta (RF - intent): 0.142

## A/B token benchmark (benchmarks/queries.yaml)
- Intent median reduction: 49.5%
- RF median reduction: 49.5%
- Regression (intent - RF): 0.0 pp
- Intent median MCP tokens: 1559
- RF median MCP tokens: 1559
