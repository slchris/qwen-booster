# scripts/

Product-facing tooling. One-off runners from the original investigation live
in `scripts/experiments/` and assume the maintainer's machine.

## Core workflow

| Script | Purpose |
| --- | --- |
| `build_opensource_suite.py` | Convert local open-benchmark exports (NIAH/LongBench/GSM8K/MMLU) into evaluated suites |
| `evaluate_api.py` | Replay a suite against an OpenAI-compatible endpoint with strict grading |
| `evaluate_long_conversation.py` | Multi-turn follow-up evaluation |
| `benchmark_decode_contexts.py` | Decode curve across context lengths; `--skip-prime` checks restore-only paths |
| `benchmark_concurrency.py` | Concurrent streaming levels with per-request and aggregate throughput |
| `render_report.py` | Measurement JSONL -> standalone offline HTML report |
| `validate_hybrid_cache.py` | GPU validation for block-aligned hybrid checkpoints |
| `validate_sampling.py` / `validate_vlm_api.py` | Sampling and VLM API consistency checks |
| `smoke_api.py` | Start/stop a local server and exercise both generation modes |

## Evidence pipeline

| Script | Purpose |
| --- | --- |
| `archive_run.py` / `archive_results.py` | Hash and freeze a run into `benchmarks/` |
| `build_*_suite.py` | Synthetic suite builders used by archived experiments |
| `audit_model_weights.py` / `audit_workload_tokens.py` | Weight and tokenizer audits |
| `summarize_*.py` | Review summaries for archived runs |
| `monitor_memory.py` | Memory sampling during a run |
| `native_mtp_reference.py` | Cross-backend token reference generation |

## scripts/experiments/

Historical one-off runners (`run_*`, `review_*`, experiment-specific
`evaluate_*`, kernel screens, probes and traces). They contain absolute paths
from the author's machine, are kept for provenance, and are not supported
tooling.
