# RF evaluation report

TOKEN_REDUCTION_GATE: PASS median_reduction=63.6 threshold_pct=55
P90_LATENCY_GATE: PASS p90_latency_ms=118.9
ANCHOR_RETENTION_GATE: PASS retention=1.000
RF_CV_GATE: PASS overall=0.440 min_bucket=0.000

## RF benchmark (queries.yaml)
- median token reduction: 63.6%
- p90 latency: 118.9 ms
- median MCP tokens: 1075

## Anchor retention (budget_training_queries.yaml)
- retention rate: 1.000

## 5-fold CV (budget_labels.jsonl)
- overall accuracy: 0.440
- per-bucket accuracy:
  - 2000: 0.382 (n=76)
  - 3000: 0.100 (n=20)
  - 4000: 0.812 (n=16)
  - 5000: 0.771 (n=35)
  - 6000: 0.238 (n=21)
  - 8000: 0.486 (n=72)
  - 10000: 0.261 (n=23)
  - 15000: 0.000 (n=3)
