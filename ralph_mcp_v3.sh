#!/usr/bin/env bash
# Ralph loop for MCP v3 — quality gates, labeling, repo-scale retrieval.
#
# Each phase loops until validate_mcp_v3_gates.sh --phase N returns 0 (grep PASS).
# Final --all verifies every blocking gate pattern + every phase marker.
#
# Blocking gates (grep ml/reports/rf_eval.md):
#   RELEVANT_FILE_RECALL_GATE  — PASS + threshold_recall=0.70
#   TASK_SUCCESS_GATE          — PASS + threshold_task_success=0.80
#   TOKEN_REDUCTION_GATE       — PASS + threshold_pct=55
#   P90_LATENCY_GATE           — PASS
#   RF_CV_GATE                 — PASS
#   RETRIEVAL_P90_GATE         — PASS
#   LABEL_BUCKET_SPREAD        — PASS + min_buckets=3
#
# Usage:
#   ./ralph_mcp_v3.sh
#   VERIFY_ONLY=1 ./ralph_mcp_v3.sh
#   MCP_V3_PHASE=3 MAX_ATTEMPTS=10 ./ralph_mcp_v3.sh
#   START_PHASE=2 ./ralph_mcp_v3.sh
#
# Requires: pip install -e '.[dev,ml]', context-eng-ml-* on PATH (agent optional if VERIFY_ONLY).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PRD="${PRD_PATH:-$ROOT/ml/prd/mcp_v3_quality_retrieval.prd.json}"
RF_REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"
STATUS_REPORT="${MCP_V3_STATUS:-$ROOT/ml/reports/mcp_v3_status.md}"
FIX_PROMPT="${FIX_PROMPT:-$ROOT/ml/prompts/fix_mcp_v3.md}"
VALIDATE="${VALIDATE_SCRIPT:-$ROOT/ml/scripts/validate_mcp_v3_gates.sh}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
SINGLE_PHASE="${MCP_V3_PHASE:-}"
START_PHASE="${START_PHASE:-}"

export PRD_PATH="$PRD"
export RF_EVAL_REPORT="$RF_REPORT"
export MCP_V3_STATUS="$STATUS_REPORT"
export FAILED_GATES=""
export ACTIVE_PHASE_NAME=""
export ACTIVE_PHASE_MARKER=""
export MCP_V3_PHASE=""

resolve_agent_cli() {
  if command -v agent >/dev/null 2>&1; then
    echo agent
    return 0
  fi
  if command -v cursor-agent >/dev/null 2>&1; then
    echo cursor-agent
    return 0
  fi
  local win_agent="${LOCALAPPDATA:-}/cursor-agent/agent.cmd"
  if [[ -f "$win_agent" ]]; then
    echo "$win_agent"
    return 0
  fi
  return 1
}

prd_implementation_order() {
  python - "$PRD" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)
order = prd.get("north_star", {}).get("implementation_order")
if not order:
    order = sorted(p["id"] for p in prd.get("phases", []))
for pid in order:
    print(pid)
PY
}

prd_phase_meta() {
  local phase_id="$1"
  python - "$PRD" "$phase_id" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)
target = sys.argv[2]
for p in prd.get("phases", []):
    if str(p["id"]) == target or p.get("key") == target:
        print(f"{p['id']}|{p['name']}|{p.get('marker_pattern', '')}")
        break
PY
}

prd_regenerate_commands() {
  python - "$PRD" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)

for cmd in prd.get("agent_loop", {}).get("regenerate_pipeline", []):
    print(cmd)
PY
}

prd_phase_verify_commands() {
  local phase_id="$1"
  python - "$PRD" "$phase_id" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)
target = sys.argv[2]
for p in prd.get("phases", []):
    if str(p["id"]) == target or p.get("key") == target:
        for cmd in p.get("verify_commands") or []:
            print(cmd)
        break
PY
}

prd_print_gate_thresholds() {
  python - "$PRD" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)

for gate in prd.get("gates", {}).get("blocking", []):
    gid = gate.get("id", "?")
    threshold = gate.get("threshold", "")
    extra = gate.get("extra_grep", "")
    print(f"  - {gid}: {gate.get('report_line_pattern', '')} {extra}".rstrip())
PY
}

prd_gate_patterns() {
  python - "$PRD" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)

for pattern in prd.get("agent_loop", {}).get("gate_grep_patterns", []):
    print(pattern)
PY
}

run_typecheck() {
  python -m mypy src/context_eng benchmarks
}

run_rf_eval() {
  local eval_cmd
  eval_cmd="$(python - "$PRD" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fh:
    prd = json.load(fh)
print(prd.get("agent_loop", {}).get("eval_command", "context-eng-ml-eval"))
PY
)"
  $eval_cmd || true
}

run_regenerate_pipeline() {
  local cmd
  while IFS= read -r cmd; do
    cmd="${cmd//$'\r'/}"
    [[ -n "$cmd" ]] || continue
    echo "ralph_mcp_v3: $cmd"
    if [[ "$cmd" == "context-eng-ml-eval" ]]; then
      $cmd || true
    else
      $cmd
    fi
  done < <(prd_regenerate_commands)
}

run_phase_tests() {
  local phase_id="$1"
  local cmd
  while IFS= read -r cmd; do
    cmd="${cmd//$'\r'/}"
    [[ -n "$cmd" ]] || continue
    echo "ralph_mcp_v3: phase test: $cmd"
    eval "$cmd"
  done < <(prd_phase_verify_commands "$phase_id")
}

verify_phase_grep() {
  local phase_id="$1"
  export MCP_V3_PHASE="$phase_id"
  "$VALIDATE" --phase "$phase_id"
}

verify_all_gates_grep() {
  "$VALIDATE" --all
}

collect_failed_grep() {
  local phase_id="$1"
  local out=""
  if ! verify_phase_grep "$phase_id" >/tmp/ralph_mcp_v3_grep.log 2>&1; then
    out="$(grep 'grep FAIL' /tmp/ralph_mcp_v3_grep.log || true)"
    if [[ -z "$out" ]]; then
      out="$(tail -n 25 /tmp/ralph_mcp_v3_grep.log)"
    fi
  fi
  echo "$out"
}

run_agent() {
  local attempt="$1"
  local phase_id="$2"
  export MCP_V3_PHASE="$phase_id"
  export FAILED_GATES="$(collect_failed_grep "$phase_id")"

  local meta
  meta="$(prd_phase_meta "$phase_id")"
  IFS='|' read -r _ ACTIVE_PHASE_NAME ACTIVE_PHASE_MARKER <<<"$meta"
  export ACTIVE_PHASE_NAME ACTIVE_PHASE_MARKER

  local prompt agent_log="$ROOT/ml/reports/ralph_mcp_v3.stdout.log"
  prompt="$(envsubst < "$FIX_PROMPT")"

  export TERM=dumb CI=1 NO_COLOR=1 FORCE_COLOR=0
  printf '\033[?1000l\033[?1002l\033[?1003l\033[?1006l' >/dev/null 2>&1 || true

  if ! "$AGENT_CLI" -p --force --trust \
    --workspace "$ROOT" \
    --model composer-2.5 \
    --output-format text \
    -- "$prompt" \
    > >(sed -u -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/\[<[0-9;]*[Mm]//g' >"$agent_log") \
    2>"$ROOT/ml/reports/ralph_mcp_v3.stderr.log"; then
    echo "ralph_mcp_v3: agent iteration $attempt phase $phase_id failed"
    return 1
  fi

  echo "ralph_mcp_v3: agent output -> $agent_log"
  tail -n 8 "$agent_log" 2>/dev/null || true
  return 0
}

run_phase_loop() {
  local phase_id
  phase_id="$(printf '%s' "$1" | tr -d '\r')"
  local attempt=0
  local meta
  meta="$(prd_phase_meta "$phase_id")"
  IFS='|' read -r _ phase_name phase_marker <<<"$meta"

  echo ""
  echo "ralph_mcp_v3: ===== Phase $phase_id — $phase_name ====="
  echo "ralph_mcp_v3: phase marker: $phase_marker"

  while true; do
    attempt=$((attempt + 1))
    echo "ralph_mcp_v3: phase $phase_id iteration $attempt / $MAX_ATTEMPTS"

    if [[ "$VERIFY_ONLY" != "1" ]]; then
      if (( phase_id >= 3 )); then
        run_regenerate_pipeline
        run_rf_eval
      fi
      run_phase_tests "$phase_id" || true
    fi

    if verify_phase_grep "$phase_id"; then
      if [[ "$VERIFY_ONLY" == "1" ]] || run_typecheck; then
        echo "ralph_mcp_v3: phase $phase_id all grep checks PASS (exit 0)"
        return 0
      fi
      echo "ralph_mcp_v3: phase $phase_id grep PASS but typecheck failed"
    else
      echo "ralph_mcp_v3: phase $phase_id grep verification failed"
      collect_failed_grep "$phase_id" | sed 's/^/  /' || true
    fi

    if [[ "$VERIFY_ONLY" == "1" ]]; then
      echo "ralph_mcp_v3: VERIFY_ONLY=1 — stopping phase $phase_id"
      return 1
    fi

    if (( attempt >= MAX_ATTEMPTS )); then
      echo "ralph_mcp_v3: stopped after $MAX_ATTEMPTS iterations on phase $phase_id"
      return 1
    fi

    run_agent "$attempt" "$phase_id" || continue
  done
}

# --- bootstrap ---
test -s "$PRD" || { echo "ralph_mcp_v3: PRD not found: $PRD"; exit 1; }
test -s "$VALIDATE" || { echo "ralph_mcp_v3: validate script missing: $VALIDATE"; exit 1; }
test -s "$FIX_PROMPT" || { echo "ralph_mcp_v3: fix prompt missing: $FIX_PROMPT"; exit 1; }

if [[ ! -f "$STATUS_REPORT" ]]; then
  mkdir -p "$(dirname "$STATUS_REPORT")"
  cp "$ROOT/ml/reports/mcp_v3_status.md" "$STATUS_REPORT" 2>/dev/null || cat >"$STATUS_REPORT" <<'EOF'
# MCP v3 phase status
PHASE1_MANIFEST_RIPGREP: PENDING
PHASE2_QUALITY_MODULE: PENDING
PHASE3_BLOCKING_GATES: PENDING
PHASE4_QUALITY_LABELS: PENDING
PHASE5_RF_RETRAIN: PENDING
EOF
fi

if [[ "$VERIFY_ONLY" != "1" ]]; then
  python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    || { echo "install deps: pip install -e '.[dev,ml]'"; exit 1; }
  AGENT_CLI="$(resolve_agent_cli)" || { echo "ralph_mcp_v3: agent CLI not found (set VERIFY_ONLY=1 for grep-only)"; exit 1; }
fi

echo "ralph_mcp_v3: PRD=$PRD"
echo "ralph_mcp_v3: report=$RF_REPORT"
echo "ralph_mcp_v3: status=$STATUS_REPORT"
echo "ralph_mcp_v3: validate=$VALIDATE"
echo "ralph_mcp_v3: blocking gates (grep):"
prd_print_gate_thresholds
echo "ralph_mcp_v3: cumulative grep patterns:"
prd_gate_patterns | sed 's/^/  - /'

strip_cr() { printf '%s' "$1" | tr -d '\r'; }

mapfile -t PHASE_ORDER < <(prd_implementation_order | tr -d '\r')
PHASES_TO_RUN=()

if [[ -n "$SINGLE_PHASE" ]]; then
  PHASES_TO_RUN=("$(strip_cr "$SINGLE_PHASE")")
elif [[ -n "$START_PHASE" ]]; then
  started=0
  start_id="$(strip_cr "$START_PHASE")"
  for pid in "${PHASE_ORDER[@]}"; do
    pid="$(strip_cr "$pid")"
    if [[ "$pid" == "$start_id" ]]; then
      started=1
    fi
    if (( started == 1 )); then
      PHASES_TO_RUN+=("$pid")
    fi
  done
  if ((${#PHASES_TO_RUN[@]} == 0)); then
    echo "ralph_mcp_v3: START_PHASE=$START_PHASE not in implementation order"
    exit 1
  fi
else
  for pid in "${PHASE_ORDER[@]}"; do
    PHASES_TO_RUN+=("$(strip_cr "$pid")")
  done
fi

for phase_id in "${PHASES_TO_RUN[@]}"; do
  phase_id="$(strip_cr "$phase_id")"
  if ! run_phase_loop "$phase_id"; then
    exit 1
  fi
done

echo ""
echo "ralph_mcp_v3: final grep verification (--all gates + phase markers)"
if ! verify_all_gates_grep; then
  echo "ralph_mcp_v3: final --all grep failed"
  "$VALIDATE" --all || true
  exit 1
fi

if [[ "$VERIFY_ONLY" != "1" ]] && ! run_typecheck; then
  echo "ralph_mcp_v3: final typecheck failed"
  exit 1
fi

echo "ralph_mcp_v3: all phases and gates verified via grep — exit 0"
exit 0
