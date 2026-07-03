# Fix MCP v3 quality / retrieval refactor (phase ${MCP_V3_PHASE:-auto})

Read:
- `ml/prd/mcp_v3_quality_retrieval.prd.json`
- `ml/reports/rf_eval.md`
- `ml/reports/mcp_v3_status.md`

## Failed grep checks
${FAILED_GATES}

## Active phase
${ACTIVE_PHASE_NAME}

Phase marker required in `ml/reports/mcp_v3_status.md`:
`${ACTIVE_PHASE_MARKER}`

## Blocking gates (grep in rf_eval.md — all must show PASS)

| Gate | Grep patterns |
|------|----------------|
| RELEVANT_FILE_RECALL | `RELEVANT_FILE_RECALL_GATE: PASS` + `threshold_recall=0.70` |
| TASK_SUCCESS | `TASK_SUCCESS_GATE: PASS` + `threshold_task_success=0.80` |
| TOKEN_REDUCTION | `TOKEN_REDUCTION_GATE: PASS` + `threshold_pct=55` |
| P90_LATENCY | `P90_LATENCY_GATE: PASS` |
| RF_CV | `RF_CV_GATE: PASS` |
| RETRIEVAL_P90 | `RETRIEVAL_P90_GATE: PASS` |
| LABEL_BUCKET_SPREAD | `LABEL_BUCKET_SPREAD: PASS` + `min_buckets=3` |

No line may contain `GATE: FAIL`.

## Phase guidance

### Phase 1 — Manifest + ripgrep
- `src/context_eng/index/manifest.py`, ripgrep in `grep_retriever.py`, Python manifest fallback
- `tests/test_manifest.py`, `tests/test_grep_retriever.py`
- Append `PHASE1_MANIFEST_RIPGREP: PASS` to `mcp_v3_status.md`

### Phase 2 — Quality module
- `src/context_eng/eval/quality.py` (`relevant_file_recall`, `task_rubric_pass`)
- `ml/data/task_eval_queries.yaml`
- `tests/test_quality_eval.py`
- Append `PHASE2_QUALITY_MODULE: PASS`

### Phase 3 — Blocking gates in eval_rf
- Replace `ANCHOR_RETENTION_GATE` with `RELEVANT_FILE_RECALL_GATE` + `TASK_SUCCESS_GATE`
- `INFERRED_ANCHOR_RETENTION` audit only
- Update `ml/data/eval_targets.yaml`
- Append `PHASE3_BLOCKING_GATES: PASS`

### Phase 4 — Quality label sweep
- `sweep_one_query_quality` in `generate_labels.py`, adaptive anchor limit
- Emit `LABEL_BUCKET_SPREAD: PASS buckets=N min_buckets=3` in rf_eval.md
- Regenerate labels + retrain
- Append `PHASE4_QUALITY_LABELS: PASS`

### Phase 5 — RF cleanup + retrieval gate
- `legacy_query_length_floors=False` default in config/budget_model
- `RETRIEVAL_P90_GATE: PASS` in rf_eval.md
- Append `PHASE5_RF_RETRAIN: PASS`

## Forbidden
- Using `expected_anchors` / task rubrics at runtime (eval/labels only)
- Lowering gate thresholds without user approval

## Verify
```
context-eng-ml-labels && context-eng-ml-train && context-eng-ml-eval || true
./ml/scripts/validate_mcp_v3_gates.sh --phase ${MCP_V3_PHASE:-1}
pytest -m "not benchmark"
python -m mypy src/context_eng benchmarks
```

Paste all gate lines from `rf_eval.md` and the phase marker you added.
