#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"

test -s "$REPORT" || { echo "missing RF eval report: $REPORT"; exit 1; }
grep -q 'ANCHOR_RETENTION_GATE: PASS' "$REPORT" || {
  echo "anchor retention gate not PASS in $REPORT"
  exit 1
}
echo "validate_anchor_retention: PASS"
