# One-time setup: install context-eng and register it globally in Cursor.
# Usage: .\scripts\install.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "Context Engineering MCP — install" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    python -m venv (Join-Path $ProjectRoot ".venv")
}

Write-Host "Installing package..."
& $VenvPython -m pip install -e "${ProjectRoot}[dev,tokens]" -q

$CursorDir = Join-Path $env:USERPROFILE ".cursor"
$CommandsDir = Join-Path $CursorDir "commands"
$CommandSrc = Join-Path $ProjectRoot ".cursor\commands\context.md"

New-Item -ItemType Directory -Force -Path $CommandsDir | Out-Null
Copy-Item -Force $CommandSrc (Join-Path $CommandsDir "context.md")
Write-Host "Copied /context command to $CommandsDir"

& $VenvPython (Join-Path $ProjectRoot "scripts\register_cursor_mcp.py") $VenvPython

Write-Host ""
Write-Host "Done. Restart Cursor (or reload MCP servers), then type:" -ForegroundColor Green
Write-Host "  /context how does auth middleware validate tokens?"
Write-Host ""
Write-Host "MCP config: $(Join-Path $CursorDir 'mcp.json')"
