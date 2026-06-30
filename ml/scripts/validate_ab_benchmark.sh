#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"

test -s "$REPORT" || { echo "missing RF eval report: $REPORT"; exit 1; }
grep -q 'RF_AB_BENCHMARK_GATE: PASS' "$REPORT" || {
  echo "A/B benchmark gate not PASS in $REPORT"
  exit 1
}
echo "validate_ab_benchmark: PASS"
exit 0
