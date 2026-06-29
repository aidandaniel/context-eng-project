#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUERIES="$ROOT/ml/data/budget_training_queries.yaml"
LABELS="$ROOT/ml/data/budget_labels.jsonl"
MIN_TOTAL="${MIN_TOTAL:-180}"
MIN_PER_BUCKET="${MIN_PER_BUCKET:-20}"

test -s "$QUERIES" || { echo "missing queries: $QUERIES"; exit 1; }
test -s "$LABELS" || { echo "missing labels: $LABELS"; exit 1; }

total=$(grep -Ec '^[[:space:]]*- id:' "$QUERIES")
if (( total < MIN_TOTAL )); then
  echo "need $MIN_TOTAL queries, have $total"
  exit 1
fi

for bucket in 2000 3000 4000 5000 6000 8000 10000 12000 15000; do
  count=$(grep -Ec "\"y\": ${bucket}(,|})" "$LABELS" || true)
  if (( count < MIN_PER_BUCKET )); then
    echo "bucket $bucket: need $MIN_PER_BUCKET, have $count"
    exit 1
  fi
done

if ! grep -q '"expected_tokens"' "$LABELS"; then
  echo 'labels missing "expected_tokens" field'
  exit 1
fi

echo "corpus OK: $total queries, all buckets >= $MIN_PER_BUCKET"
exit 0
