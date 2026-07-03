#!/usr/bin/env bash
# Grep-verify MCP v3 quality/retrieval gates + per-phase markers (PRD-driven).
#
# Usage:
#   ./ml/scripts/validate_mcp_v3_gates.sh              # first incomplete phase
#   ./ml/scripts/validate_mcp_v3_gates.sh --phase 3
#   ./ml/scripts/validate_mcp_v3_gates.sh --all        # all phase markers + all blocking gates
#   ./ml/scripts/validate_mcp_v3_gates.sh --gates-only # cumulative gates only (no phase markers)
#
# Exit 0 when every required pattern matches; exit 1 otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRD="${PRD_PATH:-$ROOT/ml/prd/mcp_v3_quality_retrieval.prd.json}"
REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"
STATUS="${MCP_V3_STATUS:-$ROOT/ml/reports/mcp_v3_status.md}"
MODE="${1:-}"
PHASE_ARG="${MCP_V3_PHASE:-}"

if [[ "$MODE" == "--phase" ]]; then
  PHASE_ARG="${2:?usage: $0 --phase <id>}"
  MODE="phase"
elif [[ "$MODE" == "--all" ]]; then
  MODE="all"
elif [[ "$MODE" == "--gates-only" ]]; then
  MODE="gates_only"
elif [[ -n "$MODE" ]]; then
  echo "usage: $0 [--phase <id>|--all|--gates-only]"
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

phases = sorted(prd.get("phases", []), key=lambda p: p["id"])
phases_by_id = {p["id"]: p for p in phases}
impl_order = prd.get("north_star", {}).get("implementation_order") or [p["id"] for p in phases]
cumulative = prd.get("agent_loop", {}).get("gate_grep_patterns", [])

def emit_status_marker(phase: dict) -> None:
    gv = phase.get("grep_verify") or {}
    for pat in gv.get("status_md") or [phase.get("marker_pattern", "")]:
        if pat:
            print(f"{status_path}\t{pat}")

def emit_phase_rf_patterns(phase: dict) -> None:
    gv = phase.get("grep_verify") or {}
    for pat in gv.get("rf_eval_md") or []:
        if pat:
            print(f"{rf_report}\t{pat}")
    for pat in phase.get("gate_grep_patterns") or []:
        print(f"{rf_report}\t{pat}")

def patterns_up_to_phase(target_id: int) -> None:
    seen_rf: set[str] = set()
    for pid in impl_order:
        phase = phases_by_id[pid]
        emit_status_marker(phase)
        for pat in (phase.get("gate_grep_patterns") or []):
            if pat not in seen_rf:
                seen_rf.add(pat)
                print(f"{rf_report}\t{pat}")
        gv = phase.get("grep_verify") or {}
        for pat in gv.get("rf_eval_md") or []:
            if pat not in seen_rf:
                seen_rf.add(pat)
                print(f"{rf_report}\t{pat}")
        if pid == target_id:
            break

if mode == "gates_only":
    for pat in cumulative:
        print(f"{rf_report}\t{pat}")
elif mode == "all":
    for phase in phases:
        emit_status_marker(phase)
    for pat in cumulative:
        print(f"{rf_report}\t{pat}")
elif mode == "phase" and phase_arg:
    try:
        target = int(phase_arg)
    except ValueError:
        target = None
        for p in phases:
            if p.get("key") == phase_arg or str(p["id"]) == phase_arg:
                target = p["id"]
                break
        if target is None:
            raise SystemExit(f"unknown phase: {phase_arg}")
    patterns_up_to_phase(target)
else:
    status_text = Path(status_path).read_text(encoding="utf-8") if Path(status_path).is_file() else ""
    target = None
    for pid in impl_order:
        marker = phases_by_id[pid].get("marker_pattern", "")
        if marker and marker not in status_text:
            target = pid
            break
    if target is None:
        for phase in phases:
            emit_status_marker(phase)
        for pat in cumulative:
            print(f"{rf_report}\t{pat}")
    else:
        patterns_up_to_phase(target)
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
  echo "validate_mcp_v3_gates: all grep checks PASS (mode=${MODE:-auto} phase=${PHASE_ARG:-auto})"
  exit 0
fi

echo "validate_mcp_v3_gates: one or more grep checks FAILED"
exit 1
