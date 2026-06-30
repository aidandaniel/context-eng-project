#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

context-eng-ml-eval || true

./ml/scripts/validate_cv.sh
./ml/scripts/validate_anchor_recall.sh
./ml/scripts/validate_ab_benchmark.sh
echo "validate_rf_gates: all PASS"
exit 0
