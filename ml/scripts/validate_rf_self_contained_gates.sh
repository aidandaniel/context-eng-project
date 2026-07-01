#!/usr/bin/env bash
# Validate all blocking gates from ml/prd/rf_only_self_contained.prd.json via grep.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRD="${PRD_PATH:-$ROOT/ml/prd/rf_only_self_contained.prd.json}"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"

test -s "$PRD" || { echo "missing PRD: $PRD"; exit 1; }
test -s "$REPORT" || { echo "missing RF eval report: $REPORT"; exit 1; }

mapfile -t GATE_PATTERNS < <(
  python - "$PRD" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)

patterns = prd.get("agent_loop", {}).get("gate_grep_patterns")
if not patterns:
    patterns = [
        g["report_line_pattern"]
        for g in prd.get("gates", {}).get("blocking", [])
        if g.get("report_line_pattern")
    ]

for pattern in patterns:
    print(pattern)
PY
)

if ((${#GATE_PATTERNS[@]} == 0)); then
  echo "no gate patterns found in PRD: $PRD"
  exit 1
fi

fail=0
for pattern in "${GATE_PATTERNS[@]}"; do
  pattern="${pattern//$'\r'/}"
  if grep -qF "$pattern" "$REPORT"; then
    echo "grep PASS: $pattern"
  else
    echo "grep FAIL: pattern not found in $REPORT -> $pattern"
    fail=1
  fi
done

if grep -q 'GATE: FAIL' "$REPORT"; then
  echo "grep FAIL: report contains GATE: FAIL"
  grep 'GATE: FAIL' "$REPORT" || true
  fail=1
fi

if (( fail != 0 )); then
  exit 1
fi

echo "validate_rf_self_contained_gates: all PASS (${#GATE_PATTERNS[@]} gates)"
exit 0
