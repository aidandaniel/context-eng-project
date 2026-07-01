#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"

test -s "$REPORT" || { echo "missing RF eval report: $REPORT"; exit 1; }
grep -q 'TOKEN_REDUCTION_GATE: PASS' "$REPORT" || {
  echo "token reduction gate not PASS in $REPORT"
  exit 1
}
echo "validate_token_reduction: PASS"
