# qwen-booster

**Measured inference experiments and reusable primitives for hybrid-attention
Qwen models on Apple Silicon.**

Validated on an M3 Max (40-core GPU, 128 GB) with Qwen3.8-27B and
Qwen3.8-Flash-Next. Every performance claim in this repository is backed by a
frozen, hashed archive under [`benchmarks/`](benchmarks/), and failed
experiments are kept on purpose.

[中文说明](README.zh-CN.md) · [Experiment log (中文，完整历史)](docs/EXPERIMENTS.zh-CN.md) ·
[Serving runbook](docs/SERVING.md)

---

## What this is

- **A measured toolkit**: long-context decode curves, concurrency benchmarks,
  cache-correctness validation, deterministic graders for open suites, and a
  zero-dependency HTML report renderer.
- **Reusable primitives** you can lift into other MLX projects: a vLLM-style
  block-aligned hybrid prefix cache, an APFS-aware checkpoint disk store, and
  strict answer graders.
- **An honest record** of what works and what does not, including retained
  failures (see the status matrix below).

## What this is not

- **Not a serving framework.** The serving path is an experiment layer over
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) pinned to a known revision.
- **Not a universal speedup.** Several optimizations are workload-specific,
  default off, and bounded by single observations.
- **Not a hardware matrix.** Two model profiles on one machine class.

## Status

| Area | State | Evidence |
| --- | --- | --- |
| Block-aligned hybrid cache (CPU + 27B GPU) | ✅ validated | [hybrid-cache-native-v1](benchmarks/hybrid-cache-native-v1/manifest.json) |
| APFS checkpoint disk store | ✅ CPU tests | `tests/test_apfs_store.py` |
| Long-context decode (Flash, warm) | ⚠️ bounded | 24.0 / 19.2 / 12.5–13.4 tok/s at 32K / 100K / 200K | 
| Concurrency (continuous batching) | ⚠️ bounded | [flash-concurrency-v1](benchmarks/flash-concurrency-v1/manifest.json) |
| 27B + DFlash2 draft | ⚠️ bounded (1.77x @32K) | [long-context-decode-scaling-v1](benchmarks/long-context-decode-scaling-v1/manifest.json) |
| 200K warm reuse | ⚠️ recipe-bound | [flash-200k-disk-spill-accept-v1](benchmarks/flash-200k-disk-spill-accept-v1/manifest.json) |
| Immediate-follow-up race fix | 🧪 candidate, causality unproven | [flash-200k-restore-race-candidate-v1](benchmarks/flash-200k-restore-race-candidate-v1/manifest.json) |
| KV quantization on Flash-Next | ❌ rejected upstream | QSA + continuous batching |
| 30 tok/s single-stream at 100K/200K | ❌ not reached | needs kernel-level work |

Warm decode on M3 Max 128 GB, Qwen3.8-Flash-Next 4-bit, greedy:

| Context | Decode | Time to first token (warm) |
| ---: | ---: | ---: |
| 32K | 24.0 tok/s | 0.3 s |
| 100K | 19.2 tok/s | 0.6 s |
| 200K | 12.5–13.4 tok/s | 2–5 s (with the 200K recipe) |

First touch of a 200K document is a batch job (~5–12.5 minutes on Flash), not
an interactive action. Details and caveats: [docs/SERVING.md](docs/SERVING.md).

## Quickstart (core, no model download)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock
uv pip install --no-deps -e .

python -m pytest -q                      # 252 CPU tests, a few seconds

# Reusable primitives and tools:
python -c "from qwen_booster.hybrid_cache import HybridPrefixCache"
python -c "from qwen_booster.grading import grade_answer, extract_number"

# Render a measurement JSONL as a standalone HTML report:
PYTHONPATH=src python scripts/render_report.py results/bench.jsonl --output report.html

# Convert local open-benchmark exports (JSONL) into evaluated suites:
PYTHONPATH=src python scripts/build_opensource_suite.py --dataset mmlu \
  --input exports/mmlu.jsonl --output results/suite-mmlu.jsonl
```

Serving needs the separate pinned environment and local model weights:

```bash
uv venv .cache/vlm-env --python 3.12
uv pip install --python .cache/vlm-env/bin/python -r requirements-vlm.lock
# see docs/SERVING.md for model preparation and the verified 32K/100K/200K configs
```

## Reusable primitives

| Module | Purpose |
| --- | --- |
| `qwen_booster.hybrid_cache` | Chained block hashes, aligned checkpoints, bounded replay, LRU — for models mixing attention KV with recurrent/GDN state |
| `qwen_booster.apfs` | Checkpoint persistence with atomic rename and APFS `clonefile` snapshots |
| `qwen_booster.grading` | Deterministic graders: exact text, JSON, choice, numeric, contains, regex |
| `qwen_booster.runtime` | `Engine` / `HybridEngine` for MLX-LM single-request execution |
| `scripts/render_report.py` | Offline HTML report from measurement JSONL |

## Repository layout

```
src/qwen_booster/   core modules and the native (MLX-LM) server
scripts/            product-facing tooling (benchmarks, evaluators, reports)
scripts/experiments one-off runners from the original investigation
tests/              252 CPU tests, no GPU required
benchmarks/         frozen evidence archives (manifests + raw outputs + logs)
docs/               serving runbook, full experiment log, release checklist
```

## Evidence

`benchmarks/<name>/manifest.json` records status, notes, artifact hashes and
archive-time source hashes. Raw JSONL, server logs and (for older archives) a
frozen `src/` snapshot are stored next to it. Read the manifest of an
experiment before citing its numbers.

## Known gaps

- Kernel/compile-level work is the remaining path to 30 tok/s at 100K+; no
  ready-made switch exists in the pinned stack.
- Tools/function-calling and session endpoints exist upstream but are not yet
  validated end-to-end here.
- Sustained reliability (soak) and multi-user concurrency beyond short tests
  are not covered.
- Publishing checklist (evidence extraction before a public push):
  [docs/RELEASE-CHECKLIST.md](docs/RELEASE-CHECKLIST.md).

## License

MIT — see [LICENSE](LICENSE). Third-party software and model licenses:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
