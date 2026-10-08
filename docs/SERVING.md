# Serving runbook (Apple Silicon, single model per machine)

This is the operational summary of the validated serving configurations on an
M3 Max (128 GB). Numbers come from the archives linked in the root README.
Everything here is bounded evidence, not a capacity plan.

## 0. Hard constraints

- **One model at a time.** Flash-Next peaks near 105 GB, Qwen3.8-27B near
  85 GB; they cannot be resident together in 128 GB.
- **Serving uses the pinned MLX-VLM environment** (`requirements-vlm.lock`),
  not the core environment. Prepare local weights first (below).
- **First touch of a 200K document is a batch job** (5–12.5 minutes). Do not
  present it as interactive.
- Servers must run one at a time; do not start benchmark runs while serving.

## 1. Preparation

Targets (download separately; no weights ship with this repo):

- Flash-Next 4-bit plus the external-PLE view (see the repository's earlier
  setup notes in `docs/EXPERIMENTS.zh-CN.md`, section "Flash 当前启动命令").
- MTP draft for Flash-Next, and/or `z-lab/Qwen3.8-27B-DFlash2` for the 27B.
- 27B 4-bit snapshot for the native path.

Environment:

```bash
uv venv .cache/vlm-env --python 3.12
uv pip install --python .cache/vlm-env/bin/python -r requirements-vlm.lock
```

## 2. Configurations

### 32K chat — plain continuous batching (recommended default)

Warm decode 24 tok/s, TTFT ~0.3 s with prefix reuse.

```bash
PYTHONPATH=src MLX_GDN_PACKED=1 .cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model .cache/models/qwen3.8-flash-next-4bit-external-ple \
  --context 32768 --port 18081 --host 127.0.0.1 --api-key <token> \
  --cache-mb 24000 --memory-mb 110000 --allocator-cache-mb 2000 \
  --token-timeout 7200 --max-num-seqs 4
```

Use `--max-num-seqs 2–4` behind a gateway with queueing. Measured aggregate
throughput at four streams is ~40 tok/s (per-stream 10–15 tok/s).

### 100K — plain APC is enough

Warm decode 19.2 tok/s, TTFT 0.6 s; first prefill ~2.7 minutes. Same command
with `--context 131072` and `--cache-mb 32000`.

### 200K — verified recipe (MTP + patch chain + disk spill)

Measured on one 204,812-token document: first request 12.5 min, follow-ups
2–5 s TTFT at ~11 tok/s, restart restore under 40 s.

```bash
env PYTHONPATH=src MLX_GDN_PACKED=1 \
  APC_CHECKPOINT_ENTRIES=4 APC_MEMORY_RESERVE_GB=3.725290298461914 \
  APC_CHECKPOINT_INTERVAL_TOKENS=0 \
  .cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model .cache/models/qwen3.8-flash-next-4bit-external-ple \
  --draft .cache/qwen3.8-flash-next-mtp-4bit --block 3 \
  --release-finished-mtp --borrow-flash-mtp-checkpoint-row \
  --credit-flash-mtp-live-warm-cache \
  --recheck-flash-mtp-cache-admission --recheck-flash-mtp-cache-restore \
  --context 262144 --cache-mb 24000 --memory-mb 110000 --allocator-cache-mb 2000 \
  --token-timeout 7200 --max-num-seqs 1 \
  --disk-cache-dir .cache/apc-flash-200k --disk-cache-mb 48000
```

Operational notes:

- Each 200K document costs one checkpoint on disk (~6 GB). Plan disk budget.
- A follow-up fired within ~0.5 s of the cold response may still recompute
  once (candidate fix is in the recipe; causality unproven). Agent clients
  should wait a few seconds after a cold response.
- MTP decodes slightly slower than the non-MTP base (11–12 vs 12.5–13.4) and
  is currently the only verified retention path.

### 27B with DFlash2 (quality path)

27B base decodes ~7.9 tok/s at 32K; with the DFlash2 draft it measured
14.0 tok/s (1.77x, 69% acceptance, token-identical greedy output).

```bash
PYTHONPATH=src MLX_GDN_PACKED=1 .cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/<rev> \
  --draft .cache/models/qwen3.8-27b-dflash2 --draft-kind dflash \
  --context 65536 --port 18082 --cache-mb 16000 --memory-mb 110000 \
  --allocator-cache-mb 2000 --token-timeout 7200 --max-num-seqs 1
```

## 3. Monitoring and control

- `GET /health` — readiness, loaded model, context limit, batching flag.
- `GET /metrics` and `/v1/metrics` — runtime metrics (management auth when
  `--api-key` is set).
- `GET /v1/cache/stats` — APC counters (`memory_skips`, `disk_hits`,
  `disk_writes`, `resident_bytes`, `last_reject`).
- `POST /v1/cache/reset` — clears in-memory APC state.

For offline result viewing use `scripts/render_report.py`; the JSONL outputs
stay the source of truth.

## 4. Things not to do

- Do not enable `--kv-bits` for Flash-Next: QSA + continuous batching rejects
  quantized KV upstream.
- Do not expect 30 tok/s single-stream at 100K/200K; that needs kernel-level
  work.
- Do not expose the server directly to a network without a reverse proxy and
  `--api-key`.
- Do not run 200K concurrency experiments with `max-num-seqs` above 1 without
  re-measuring memory headroom.
