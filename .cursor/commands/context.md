---
name: context
description: "Get budgeted codebase context. Usage: /context <your task or question>"
---

# Context Engineering

## Task: $ARGUMENTS

The **context-eng** MCP server is enabled. Do not read whole files first.

1. Call `prepare_context` with `query` set to the task above (omit `workspace_root`
   unless the MCP is registered globally and cwd is wrong).
2. Use only the returned `formatted_context` and `bundle` chunks to answer or
   continue the task.
3. If something important is missing, call `expand_context` with the bundle id —
   do not bulk-read files.

Report the bundle id, token budget used, and which files were included.
