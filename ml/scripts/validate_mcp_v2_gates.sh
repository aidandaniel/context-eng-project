#!/usr/bin/env bash
# Grep-verify MCP v2 cumulative gates + per-phase markers (PRD-driven).
#
# Usage:
#   ./ml/scripts/validate_mcp_v2_gates.sh              # first incomplete phase
#   ./ml/scripts/validate_mcp_v2_gates.sh --phase 2    # phase 2 only
#   MCP_V2_PHASE=3 ./ml/scripts/validate_mcp_v2_gates.sh
#   ./ml/scripts/validate_mcp_v2_gates.sh --all        # every phase marker + gates
#
# Exit 0 when all required grep patterns match; exit 1 otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRD="${PRD_PATH:-$ROOT/ml/prd/mcp_v2_roadmap.prd.json}"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"
STATUS="${MCP_V2_STATUS:-$ROOT/ml/reports/mcp_v2_status.md}"
MODE="${1:-}"
PHASE_ARG="${MCP_V2_PHASE:-}"

if [[ "$MODE" == "--phase" ]]; then
  PHASE_ARG="${2:?usage: $0 --phase <id>}"
  MODE="phase"
elif [[ "$MODE" == "--all" ]]; then
  MODE="all"
elif [[ -n "$MODE" ]]; then
  echo "usage: $0 [--phase <id>|--all]"
  exit 2
fi

test -s "$PRD" || { echo "missing PRD: $PRD"; exit 1; }

grep_file() {
  local file="$1"
  local pattern="$2"
  pattern="${pattern//$'\r'/}"
  if [[ ! -s "$file" ]]; then
    echo "grep FAIL: missing file $file -> $pattern"
    return 1
  fi
  if grep -qF "$pattern" "$file"; then
    echo "grep PASS [$file]: $pattern"
    return 0
  fi
  echo "grep FAIL [$file]: $pattern"
  return 1
}

run_grep_checks() {
  local fail=0
  local line file pattern
  while IFS=$'\t' read -r file pattern; do
    [[ -n "$pattern" ]] || continue
    if ! grep_file "$file" "$pattern"; then
      fail=1
    fi
  done < <(
    python - "$PRD" "$REPORT" "$STATUS" "$MODE" "$PHASE_ARG" <<'PY'
import json
import sys
from pathlib import Path

prd_path, rf_report, status_path, mode, phase_arg = sys.argv[1:6]
with open(prd_path, encoding="utf-8") as fh:
    prd = json.load(fh)

phases_by_id = {p["id"]: p for p in prd.get("phases", [])}
impl_order = prd.get("north_star", {}).get("implementation_order") or sorted(phases_by_id)
cumulative = prd.get("agent_loop", {}).get("gate_grep_patterns", [])

def phase_patterns(phase_id: int) -> None:
    phase = phases_by_id.get(phase_id)
    if not phase:
        return
    gv = phase.get("grep_verify") or {}
    for pat in gv.get("status_md") or [phase.get("marker_pattern", "")]:
        if pat:
            print(f"{status_path}\t{pat}")
    for pat in gv.get("rf_eval_md") or []:
        if pat:
            print(f"{rf_report}\t{pat}")

if mode == "all":
    for pid in sorted(phases_by_id):
        phase_patterns(pid)
    for pat in cumulative:
        print(f"{rf_report}\t{pat}")
elif mode == "phase" and phase_arg:
    try:
        target = int(phase_arg)
    except ValueError:
        target = None
        for p in prd.get("phases", []):
            if p.get("key") == phase_arg or str(p["id"]) == phase_arg:
                target = p["id"]
                break
        if target is None:
            raise SystemExit(f"unknown phase: {phase_arg}")
    # Phases up to and including target (in implementation order) must grep PASS.
    for pid in impl_order:
        phase_patterns(pid)
        if pid == target:
            break
    for pat in cumulative:
        print(f"{rf_report}\t{pat}")
else:
    # First incomplete phase (by implementation order).
    status_text = Path(status_path).read_text(encoding="utf-8") if Path(status_path).is_file() else ""
    target = None
    for pid in impl_order:
        phase = phases_by_id[pid]
        marker = phase.get("marker_pattern", "")
        if marker and marker not in status_text:
            target = pid
            break
    if target is None:
        for pid in sorted(phases_by_id):
            phase_patterns(pid)
        for pat in cumulative:
            print(f"{rf_report}\t{pat}")
    else:
        for pid in impl_order:
            phase_patterns(pid)
            if pid == target:
                break
        for pat in cumulative:
            print(f"{rf_report}\t{pat}")
PY
  )

  if [[ -s "$REPORT" ]] && grep -q 'GATE: FAIL' "$REPORT"; then
    echo "grep FAIL: $REPORT contains GATE: FAIL"
    grep 'GATE: FAIL' "$REPORT" || true
    fail=1
  fi

  return "$fail"
}

if run_grep_checks; then
  echo "validate_mcp_v2_gates: all grep checks PASS (mode=${MODE:-auto} phase=${PHASE_ARG:-auto})"
  exit 0
fi

echo "validate_mcp_v2_gates: one or more grep checks FAILED"
exit 1
