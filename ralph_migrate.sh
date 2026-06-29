#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

QUERIES="$ROOT/ml/data/budget_training_queries.yaml"
LABELS="$ROOT/ml/data/budget_labels.jsonl"
PROMPT_TEMPLATE="$ROOT/ml/prompts/generate_training_query.md"

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

python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
  || { echo "install deps: pip install -e '.[dev,ml]'"; exit 1; }

AGENT_CLI="$(resolve_agent_cli)" || { echo "agent CLI not found on PATH"; exit 1; }

attempt=0
while true; do
  remaining="$(count_remaining)"
  echo "ralph_migrate: $remaining queries/bucket-slots remaining"

  if (( remaining == 0 )); then
    if run_typecheck; then
      ./ml/scripts/validate_corpus.sh
      context-eng-ml-train
      echo "ralph_migrate: done"
      exit 0
    fi
    echo "ralph_migrate: corpus complete but typecheck failed; spawning agent to fix"
  fi

  attempt=$((attempt + 1))
  if (( attempt > MAX_ATTEMPTS )); then
    echo "ralph_migrate: stopped after $MAX_ATTEMPTS iterations (remaining=$remaining)"
    exit 1
  fi

  eval "$(python ml/scripts/corpus_gap.py --export-shell)"
  prompt="$(envsubst < "$PROMPT_TEMPLATE")"

  # Avoid mouse/ANSI tracking noise in logs (common on Windows Git Bash).
  export TERM=dumb CI=1 NO_COLOR=1 FORCE_COLOR=0

  "$AGENT_CLI" -p --force --trust \
    --workspace "$ROOT" \
    --model composer-2.5 \
    --output-format text \
    -- "$prompt" 2>"$ROOT/ml/reports/ralph_agent.stderr.log" \
    | tr -d '\033' \
    || { echo "ralph_migrate: agent iteration $attempt failed (see ml/reports/ralph_agent.stderr.log)"; continue; }

  python ml/scripts/merge_incoming_queries.py
  context-eng-ml-labels

  if ! run_typecheck; then
    echo "ralph_migrate: typecheck failed after iteration $attempt (will retry)"
  fi
done
