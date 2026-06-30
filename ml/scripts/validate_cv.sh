#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"

test -s "$REPORT" || { echo "missing RF eval report: $REPORT"; exit 1; }
grep -q 'RF_CV_GATE: PASS' "$REPORT" || { echo "CV gate not PASS in $REPORT"; exit 1; }
echo "validate_cv: PASS"
exit 0
