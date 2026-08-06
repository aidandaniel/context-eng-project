# Fixture App

A small, self-contained sample application used only by the Context Engineering
MCP benchmark. It has **no external dependencies** and is never executed in
production; its purpose is to give the benchmark a realistic codebase to gather
context from.

## Layout

- `src/auth/` - token creation, refresh/logout lifecycle, request middleware
- `src/users/` - user domain models and service
- `src/billing/` - invoices and payments (unrelated to auth)
- `src/inventory/` - product catalog and stock (pure noise)
- `src/api/` - routes and request dispatcher wiring modules together
- `src/utils/` - logging and settings helpers
- `tests/` - unit tests per module

## Why it exists

The benchmark compares two ways of feeding an LLM context for a query:

1. **Baseline** - grep the keywords and read the top matching files in full.
2. **MCP** - call `get_context_bundle` and receive ranked, budgeted slices.

A mix of relevant (`auth`) and irrelevant (`billing`, `inventory`) modules makes
the difference measurable.
