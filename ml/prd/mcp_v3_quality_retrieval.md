# MCP v3 — Quality, Labeling, Retrieval

PRD: [`mcp_v3_quality_retrieval.prd.json`](mcp_v3_quality_retrieval.prd.json)

## Ralph loop

```bash
./ralph_mcp_v3.sh
VERIFY_ONLY=1 ./ralph_mcp_v3.sh
MCP_V3_PHASE=3 ./ralph_mcp_v3.sh
./ml/scripts/validate_mcp_v3_gates.sh --all
./ml/scripts/validate_mcp_v3_gates.sh --gates-only
```

## Blocking gates (grep `ml/reports/rf_eval.md`)

| Gate | Grep patterns |
|------|----------------|
| RELEVANT_FILE_RECALL | `RELEVANT_FILE_RECALL_GATE: PASS`, `threshold_recall=0.70` |
| TASK_SUCCESS | `TASK_SUCCESS_GATE: PASS`, `threshold_task_success=0.80` |
| TOKEN_REDUCTION | `TOKEN_REDUCTION_GATE: PASS`, `threshold_pct=55` |
| P90_LATENCY | `P90_LATENCY_GATE: PASS` |
| RF_CV | `RF_CV_GATE: PASS` |
| RETRIEVAL_P90 | `RETRIEVAL_P90_GATE: PASS` |
| LABEL_BUCKET_SPREAD | `LABEL_BUCKET_SPREAD: PASS`, `min_buckets=3` |

Phase markers in `ml/reports/mcp_v3_status.md`: `PHASE1_MANIFEST_RIPGREP` … `PHASE5_RF_RETRAIN`.

## Phases

1. Manifest + ripgrep retriever  
2. Quality eval module + task_eval corpus  
3. Blocking gates in eval_rf  
4. Quality-aware labels + bucket spread  
5. RF retrain + retrieval latency gate  
