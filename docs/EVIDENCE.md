# Evidence bundles

The source repository intentionally does **not** track `benchmarks/`. That
directory holds roughly 200 MB across ~7,000 raw experiment files (JSONL
outputs, server logs, manifests, frozen sources). It is distributed as a
release bundle so the source repository stays source-sized and the experiment
record stays immutable.

After extracting a bundle at the repository root, every `benchmarks/...` link
in the READMEs and `docs/EXPERIMENTS.zh-CN.md` resolves.

## Download and verify

1. Open the **Releases** page of this repository and download
   `qwen-booster-evidence-vX.Y.Z.tar.gz` and its `.sha256` sidecar. No
   evidence bundle has been published until a release exists.
2. Verify the checksum:

   ```bash
   shasum -a 256 -c qwen-booster-evidence-v0.1.0.tar.gz.sha256
   ```

3. Extract at the repository root:

   ```bash
   tar -xzf qwen-booster-evidence-v0.1.0.tar.gz -C .
   ls benchmarks | head
   ```

The archive contains `benchmarks/<name>/manifest.json` plus the raw files for
each experiment. Read the manifest before citing any number: it records the
scope, status, artifact hashes and archive-time source hashes. Failed
experiments are kept on purpose.

## Build a bundle yourself

From a maintained working copy that still has `benchmarks/`:

```bash
python scripts/pack_evidence.py --source benchmarks \
  --output dist/qwen-booster-evidence-v0.1.0.tar.gz
```

The packer is deterministic (sorted paths, zeroed metadata) and writes a
`.sha256` sidecar, so identical inputs produce identical hashes.

## Run scoring locally

Archived numbers are history, not a leaderboard. To measure your own
hardware, use the supported tooling and keep raw outputs:

```bash
# 1. Convert a local open-benchmark export into a suite
PYTHONPATH=src python scripts/build_opensource_suite.py --dataset mmlu \
  --input exports/mmlu.jsonl --output results/suite-mmlu.jsonl

# 2. Replay it against a running server (strict grading, raw JSONL output)
PYTHONPATH=src python scripts/evaluate_api.py --suite results/suite-mmlu.jsonl \
  --output results/mmlu-run.jsonl --endpoint http://127.0.0.1:18081/v1 \
  --label "your model, quantization, cache state"

# 3. Decode/concurrency curves on this machine
PYTHONPATH=src python scripts/benchmark_decode_contexts.py \
  --endpoint http://127.0.0.1:18081/v1 --model <local model path> \
  --tokenizer-dir <local model path> --lengths 32768 102400 \
  --tokens 128 --output results/decode-curve.jsonl

# 4. View anything as a standalone HTML report
PYTHONPATH=src python scripts/render_report.py results/decode-curve.jsonl \
  --output results/decode-curve.html

# 5. Freeze a run (with hashes and a manifest) before comparing across time
PYTHONPATH=src python scripts/archive_run.py --destination benchmarks/<new-name> \
  --status complete --note "what was measured and what was not" \
  results/decode-curve.jsonl results/decode-curve.html
```

Rules that keep the evidence usable: never overwrite an existing output file,
never delete a failing run, always state cache state and repetition count,
and do not extrapolate short-context results to long-context claims.

## 中文说明

`benchmarks/` 证据目录**不随源码仓库发布**（约 200MB、7000 个原始文件），而是作为
Release 附件分发：在本仓库 Releases 页面下载 `qwen-booster-evidence-vX.Y.Z.tar.gz`
和 `.sha256`，校验后在仓库根目录解压即可恢复 `benchmarks/`，文档里的链接随即有效。
自建 bundle 用 `scripts/pack_evidence.py`（确定性打包，哈希可复现）。本地跑分请用
`build_opensource_suite.py` → `evaluate_api.py` / `benchmark_decode_contexts.py` →
`render_report.py`，新证据用 `archive_run.py` 冻结；不要覆盖旧输出、不要删除失败记录。
