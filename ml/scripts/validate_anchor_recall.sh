#!/usr/bin/env bash
# Legacy alias — anchor retention replaced RF_ANCHOR_RECALL_GATE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/ml/scripts/validate_anchor_retention.sh"
