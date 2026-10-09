# Changelog

## 0.1.0 - 2026-10-08

Initial public-ready snapshot. Everything below is bounded by the evidence in
`benchmarks/`; read those manifests before drawing conclusions.

### Distribution

- `benchmarks/` evidence is no longer tracked in the source repository. Release
  bundles are built deterministically with `scripts/pack_evidence.py` and
  documented in `docs/EVIDENCE.md` (download, verify, local rebuild, local
  scoring).

### Added

- Block-aligned hybrid prefix cache (`qwen_booster.hybrid_cache`) following
  vLLM automatic-prefix-caching identity: chained block hashes, aligned
  checkpoints, bounded replay, LRU, exact-aligned post-generation commits.
- APFS-aware checkpoint disk store (`qwen_booster.apfs`): safetensors via
  temp file plus atomic rename, `clonefile` copy-on-write snapshots, atomic
  index with stale-writer refresh.
- Deterministic graders: `choice`, `numeric`, `contains_all/any`, `regex`,
  plus the original `exact_text`/`json` rules.
- Offline open-suite converter (`scripts/build_opensource_suite.py`) for
  RULER-style NIAH, LongBench, GSM8K and MMLU local exports.
- Zero-dependency HTML report renderer (`scripts/render_report.py`).
- Long-context decode and concurrency benchmarks
  (`scripts/benchmark_decode_contexts.py`, `scripts/benchmark_concurrency.py`)
  with a restore-only mode.
- GPU validation runner for the hybrid cache
  (`scripts/validate_hybrid_cache.py`).
- VLM launcher options: `--host`, `--max-num-seqs`, `--api-key`,
  `--draft-kind dflash|eagle3|mtp`, `--kv-bits`, `--kv-quant-scheme`,
  `--quantized-kv-start`.
- Candidate fix for the 200K restore-admission race (`--recheck-flash-mtp-cache-restore`).
- 252 CPU tests.

### Measured summary (M3 Max 128GB)

- Flash-Next warm decode: 24.0 tok/s at 32K, 19.2 at 100K, 12.5-13.4 at 200K.
- Continuous batching: 4 concurrent streams ~40 tok/s aggregate.
- 27B + z-lab DFlash2 draft: 1.77x decode at 32K (69% draft acceptance).
- 200K usable recipe: ~12.5 min first touch per document, 2-5 s TTFT and
  ~11 tok/s afterwards, disk restore under 40 s after restart.

### Known limitations

- 200K warm caching requires the MTP + disk-spill recipe; plain APC still
  misses after a cold request.
- `--kv-bits` is rejected by upstream for Flash-Next QSA continuous batching.
- The immediate-follow-up race fix is a candidate whose causal efficacy is
  unproven.
- Serving remains single-model, local-first; two models cannot be resident
  together within 128 GB.
