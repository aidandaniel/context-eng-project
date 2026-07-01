#!/usr/bin/env bash
# Legacy alias — token reduction replaced RF_AB_BENCHMARK_GATE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/ml/scripts/validate_token_reduction.sh"
