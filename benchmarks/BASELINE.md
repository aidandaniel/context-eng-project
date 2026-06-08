# Benchmark Baseline Snapshot

Committed snapshot of the before/after token-reduction benchmark on
`benchmarks/fixture_repo` (the live report `benchmarks/results/latest.md` is
gitignored). Regenerate with:

```
context-eng-benchmark
# or: python -m benchmarks.compare
```

## Summary (14 queries)

| Metric | Value | Gate |
|--------|-------|------|
| Median token reduction | 63.6% | >= 30% PASS |
| Median anchor recall | 100% | >= 95% PASS |
| Min anchor recall | 100% | - |
| Mean supporting recall | 61% | informational |
| p90 latency | ~92 ms | < 3000 ms PASS |
| Median baseline tokens | 2,330 | - |
| Median MCP tokens | 762 | - |

## Per-query

| Query | Intent | Baseline | MCP | Reduction | Anchor | Support | Exp |
|-------|--------|---------:|----:|----------:|:------:|:-------:|:---:|
| debug_refresh_token | debug | 2,330 | 628 | 73.0% | 100% | 100% | 0 |
| debug_verify_token | debug | 2,850 | 1,775 | 37.7% | 100% | 100% | 0 |
| debug_middleware_401 | debug | 2,829 | 691 | 75.6% | 100% | 0% | 0 |
| implement_token_kind | implement | 2,927 | 1,555 | 46.9% | 100% | 100% | 0 |
| implement_invoice_discount | implement | 1,825 | 1,129 | 38.1% | 100% | 100% | 0 |
| implement_user_search | implement | 1,712 | 709 | 58.6% | 100% | 0% | 0 |
| explain_refresh_flow | explain | 2,330 | 862 | 63.0% | 100% | 50% | 0 |
| explain_dispatch | explain | 2,155 | 565 | 73.8% | 100% | 0% | 0 |
| explain_payment_charge | explain | 1,727 | 618 | 64.2% | 100% | 0% | 0 |
| refactor_rename_refresh | refactor | 2,330 | 628 | 73.0% | 100% | 100% | 0 |
| refactor_extract_signing | refactor | 3,279 | 815 | 75.1% | 100% | 100% | 0 |
| refactor_user_store | refactor | 2,633 | 1,240 | 52.9% | 100% | 0% | 0 |
| review_token_security | review | 2,927 | 1,738 | 40.6% | 100% | 100% | 0 |
| review_routes_pr | review | 2,117 | 670 | 68.4% | 100% | 100% | 0 |
| **MEDIAN** | - | 2,330 | 762 | 63.6% | 100% | 61% | 0 |

## Notes

- **Baseline** = `grep_top_k_full_files` (k=5): grep the query keywords and read
  the top matching files in full, the way an agent without context engineering
  typically gathers context.
- **MCP** = `get_context_bundle`: ranked, budgeted symbol slices + import
  neighbors + keyword snippets, with explicitly mentioned files always included.
- Token counts use the same `tiktoken` (cl100k_base) estimator for both paths.
- Supporting recall is informational and not gated. The misses are mostly
  *reverse* dependencies (files that import the anchor, e.g. tests/middleware)
  which the forward 1-hop import graph does not capture; a v2 reverse-edge index
  would raise this.
