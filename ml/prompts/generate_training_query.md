# Generate RF budget training queries

You are extending the context-eng Random Forest training corpus.

## Targets this batch
- Deficient bucket(s): ${DEFICIT_BUCKETS}
- Remaining slots to fill: ${REMAINING}
- Batch size: ${BATCH_SIZE} new YAML entries
- Workspace fixture: benchmarks/fixture_repo (paths relative to repo root)

## Hard rules
1. Write ONLY new YAML list items (do not repeat existing ids).
2. Each entry MUST have: `id`, `query`, `expected_anchors`.
3. Queries must be natural language and **path-free** (no `src/...` in query text).
4. `expected_anchors` must be real files under benchmarks/fixture_repo.
5. For bucket ${PRIMARY_BUCKET}:
   - Use ${ANCHOR_GUIDANCE} anchors from fixture domains (auth, billing, users, api, inventory, platform).
   - Set `target_budget: ${PRIMARY_BUCKET}` when sweep alone cannot hit this bucket.
6. Vary intent verbs: debug / explain / implement / refactor / review.
7. Unique ids: `{domain}_{bucket}_{nn}` (increment nn).

## Output
Write to: ml/data/incoming/batch_${TIMESTAMP}.yaml

Example entry:
```yaml
- id: billing_4000_03
  query: "Trace discount application before payment capture fails."
  expected_anchors: ["src/billing/invoice.py", "src/billing/payment.py", "tests/test_billing.py"]
  target_budget: 4000
```

## After writing YAML
Run these commands in order:
```
python ml/scripts/merge_incoming_queries.py
context-eng-ml-labels
./ml/scripts/validate_corpus.sh
```

Report: ids added, primary bucket targeted, and whether validate passed.
