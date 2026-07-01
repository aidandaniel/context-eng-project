#!/usr/bin/env bash
# Ralph loop for RF-only self-contained MCP (PRD-driven, grep gate checks).
#
# Drives an agent to fix failing implementation until all blocking gates in
# ml/prd/rf_only_self_contained.prd.json grep PASS in ml/reports/rf_eval.md.
#
# Usage:
#   ./ralph_rf_self_contained.sh
#   MAX_ATTEMPTS=10 PRD_PATH=ml/prd/rf_only_self_contained.prd.json ./ralph_rf_self_contained.sh
#
# Requires: pip install -e '.[dev,ml]', context-eng-ml-* on PATH, agent CLI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PRD="${PRD_PATH:-$ROOT/ml/prd/rf_only_self_contained.prd.json}"
RF_REPORT="${RF_EVAL_REPORT:-$ROOT/ml/reports/rf_eval.md}"
FIX_PROMPT="${FIX_PROMPT:-$ROOT/ml/prompts/fix_rf_eval.md}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"

export PRD_PATH="$PRD"
export RF_EVAL_REPORT="$RF_REPORT"
export FAILED_GATES=""

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

prd_gate_patterns() {
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
  # Eval may exit non-zero while gates are still failing; report is still useful.
  $eval_cmd || true
}

run_regenerate_pipeline() {
  local cmd
  while IFS= read -r cmd; do
    cmd="${cmd//$'\r'/}"
    [[ -n "$cmd" ]] || continue
    echo "ralph_rf_self_contained: $cmd"
    if [[ "$cmd" == "context-eng-ml-eval" ]]; then
      $cmd || true
    else
      $cmd
    fi
  done < <(prd_regenerate_commands)
}

failed_gates() {
  if [[ ! -s "$RF_REPORT" ]]; then
    echo "RF eval report missing: $RF_REPORT"
    return 0
  fi

  local pattern missing=0
  while IFS= read -r pattern; do
    pattern="${pattern//$'\r'/}"
    [[ -n "$pattern" ]] || continue
    if ! grep -qF "$pattern" "$RF_REPORT"; then
      echo "MISSING: $pattern"
      missing=1
    fi
  done < <(prd_gate_patterns)

  if grep -q 'GATE: FAIL' "$RF_REPORT"; then
    grep 'GATE: FAIL' "$RF_REPORT" || true
    missing=1
  fi

  if (( missing == 0 )); then
    return 1
  fi
  return 0
}

run_agent() {
  local attempt="$1"
  export FAILED_GATES=""
  if failed_gates; then
    FAILED_GATES="$(failed_gates)"
  fi

  local prompt agent_log="$ROOT/ml/reports/ralph_rf_self_contained.stdout.log"
  prompt="$(envsubst < "$FIX_PROMPT")"

  export TERM=dumb CI=1 NO_COLOR=1 FORCE_COLOR=0
  printf '\033[?1000l\033[?1002l\033[?1003l\033[?1006l' >/dev/null 2>&1 || true

  if ! "$AGENT_CLI" -p --force --trust \
    --workspace "$ROOT" \
    --model composer-2.5 \
    --output-format text \
    -- "$prompt" \
    > >(sed -u -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/\[<[0-9;]*[Mm]//g' >"$agent_log") \
    2>"$ROOT/ml/reports/ralph_rf_self_contained.stderr.log"; then
    echo "ralph_rf_self_contained: agent iteration $attempt failed (see ml/reports/ralph_rf_self_contained.*.log)"
    return 1
  fi

  echo "ralph_rf_self_contained: agent output -> $agent_log"
  tail -n 8 "$agent_log" 2>/dev/null || true
  return 0
}

# --- bootstrap ---
test -s "$PRD" || { echo "ralph_rf_self_contained: PRD not found: $PRD"; exit 1; }
test -s "$FIX_PROMPT" || { echo "ralph_rf_self_contained: fix prompt not found: $FIX_PROMPT"; exit 1; }

python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
  || { echo "install deps: pip install -e '.[dev,ml]'"; exit 1; }

AGENT_CLI="$(resolve_agent_cli)" || { echo "ralph_rf_self_contained: agent CLI not found on PATH"; exit 1; }

echo "ralph_rf_self_contained: PRD=$PRD"
echo "ralph_rf_self_contained: report=$RF_REPORT"
echo "ralph_rf_self_contained: gates (grep patterns):"
prd_gate_patterns | sed 's/^/  - /'

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "ralph_rf_self_contained: iteration $attempt / $MAX_ATTEMPTS"

  run_regenerate_pipeline
  run_rf_eval

  if ./ml/scripts/validate_rf_self_contained_gates.sh; then
    if run_typecheck; then
      echo "ralph_rf_self_contained: all PRD gates grep PASS + typecheck OK"
      exit 0
    fi
    echo "ralph_rf_self_contained: gates PASS but typecheck failed"
  else
    echo "ralph_rf_self_contained: gate validation failed (grep)"
    failed_gates || true
  fi

  if (( attempt >= MAX_ATTEMPTS )); then
    echo "ralph_rf_self_contained: stopped after $MAX_ATTEMPTS iterations"
    exit 1
  fi

  run_agent "$attempt" || continue
done
