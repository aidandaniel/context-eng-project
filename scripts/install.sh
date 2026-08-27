#!/usr/bin/env bash
# One-time setup: install context-eng and register it in ~/.cursor/mcp.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"

echo "Context Engineering MCP — install"
echo "Project: $ROOT"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv "$ROOT/.venv"
fi

echo "Installing package..."
"$VENV_PY" -m pip install -e "${ROOT}[dev,tokens]" -q

COMMANDS_DIR="${HOME}/.cursor/commands"
mkdir -p "$COMMANDS_DIR"
cp -f "$ROOT/.cursor/commands/context.md" "$COMMANDS_DIR/context.md"
echo "Copied /context command to $COMMANDS_DIR"

"$VENV_PY" "$ROOT/scripts/register_cursor_mcp.py" "$VENV_PY"

echo
echo "Done. Restart Cursor (or reload MCP servers), then type:"
echo "  /context how does auth middleware validate tokens?"
