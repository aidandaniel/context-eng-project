#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

QUERIES="$ROOT/ml/data/budget_training_queries.yaml"
LABELS="$ROOT/ml/data/budget_labels.jsonl"
RF_REPORT="$ROOT/ml/reports/rf_eval.md"
CORPUS_PROMPT="$ROOT/ml/prompts/generate_training_query.md"
RF_PROMPT="$ROOT/ml/prompts/fix_rf_eval.md"

MIN_TOTAL="${MIN_TOTAL:-180}"
MIN_PER_BUCKET="${MIN_PER_BUCKET:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"

export MIN_TOTAL MIN_PER_BUCKET

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

run_typecheck() {
  python -m mypy src/context_eng benchmarks
}

count_remaining() {
  local remaining=0 total count bucket

  if [[ -f "$QUERIES" ]]; then
    total=$(grep -Ec '^[[:space:]]*- id:' "$QUERIES" || true)
  else
    total=0
  fi
  if (( total < MIN_TOTAL )); then
    remaining=$((remaining + MIN_TOTAL - total))
  fi

  for bucket in 2000 3000 4000 5000 6000 8000 10000 12000 15000; do
    if [[ -f "$LABELS" ]]; then
      count=$(grep -Ec "\"y\": ${bucket}(,|})" "$LABELS" || true)
    else
      count=0
    fi
    if (( count < MIN_PER_BUCKET )); then
      remaining=$((remaining + MIN_PER_BUCKET - count))
    fi
  done

  if [[ ! -f "$LABELS" ]] || ! grep -q '"expected_tokens"' "$LABELS" 2>/dev/null; then
    remaining=$((remaining + 1))
  fi

  echo "$remaining"
}

run_rf_eval() {
  context-eng-ml-eval || true
}

failed_rf_gates() {
  local failed=""
  if [[ ! -f "$RF_REPORT" ]]; then
    echo "RF eval report missing"
    return 0
  fi
  grep 'GATE: FAIL' "$RF_REPORT" || true
}

check_rf_gates() {
  run_rf_eval
  local ok=0
  ./ml/scripts/validate_cv.sh && ok=$((ok + 1)) || true
  ./ml/scripts/validate_anchor_recall.sh && ok=$((ok + 1)) || true
  ./ml/scripts/validate_ab_benchmark.sh && ok=$((ok + 1)) || true
  (( ok == 3 ))
}

run_agent() {
  local prompt_file="$1"
  eval "$(python ml/scripts/corpus_gap.py --export-shell 2>/dev/null || true)"
  export FAILED_GATES="$(failed_rf_gates)"
  local prompt agent_log="$ROOT/ml/reports/ralph_agent.stdout.log"
  prompt="$(envsubst < "$prompt_file")"

  export TERM=dumb CI=1 NO_COLOR=1 FORCE_COLOR=0

  # Disable mouse tracking so Git Bash does not flood the terminal with [<x;y;zM noise.
  printf '\033[?1000l\033[?1002l\033[?1003l\033[?1006l' >/dev/null 2>&1 || true

  if ! "$AGENT_CLI" -p --force --trust \
    --workspace "$ROOT" \
    --model composer-2.5 \
    --output-format text \
    -- "$prompt" \
    > >(sed -u -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/\[<[0-9;]*[Mm]//g' >"$agent_log") \
    2>"$ROOT/ml/reports/ralph_agent.stderr.log"; then
    echo "ralph_migrate: agent iteration $attempt failed (see ml/reports/ralph_agent.*.log)"
    return 1
  fi

  echo "ralph_migrate: agent output -> ml/reports/ralph_agent.stdout.log"
  tail -n 8 "$agent_log" 2>/dev/null || true
}

python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
  || { echo "install deps: pip install -e '.[dev,ml]'"; exit 1; }

AGENT_CLI="$(resolve_agent_cli)" || { echo "agent CLI not found on PATH"; exit 1; }

attempt=0
while true; do
  remaining="$(count_remaining)"
  echo "ralph_migrate: $remaining corpus slots remaining"

  if (( remaining == 0 )) && run_typecheck && ./ml/scripts/validate_corpus.sh; then
    context-eng-ml-train
    if check_rf_gates; then
      echo "ralph_migrate: corpus + CV + anchor recall + A/B benchmark all PASS"
      exit 0
    fi
    echo "ralph_migrate: corpus complete; RF eval gates failing:"
    failed_rf_gates || true
    prompt_file="$RF_PROMPT"
  elif (( remaining == 0 )); then
    echo "ralph_migrate: corpus size OK but validate/typecheck failed"
    prompt_file="$RF_PROMPT"
  else
    prompt_file="$CORPUS_PROMPT"
  fi

  attempt=$((attempt + 1))
  if (( attempt > MAX_ATTEMPTS )); then
    echo "ralph_migrate: stopped after $MAX_ATTEMPTS iterations"
    exit 1
  fi

  run_agent "$prompt_file" || continue

  python ml/scripts/merge_incoming_queries.py
  context-eng-ml-labels

  if ! run_typecheck; then
    echo "ralph_migrate: typecheck failed after iteration $attempt"
  fi
done
