#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

context-eng-ml-eval || true
./ml/scripts/validate_rf_self_contained_gates.sh
echo "validate_rf_gates: all PASS"
exit 0
