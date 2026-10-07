"""GPU validation for block-aligned hybrid checkpoints on the native 27B path.

Full mode runs one model load across ordered cases on the same token prompt:
cold prefill, warm checkpoint reuse, changed final token, extended prefix,
no-reuse cold control, then a fresh-engine disk restore. Greedy output ids
must match the cold run exactly for warm, control and disk cases, or the
summary records a parity failure and exits nonzero. --restore-only reruns a
new process against an existing disk directory to prove cross-process restore.

This validates cache mechanics only: it is not throughput, quality or
daily-use acceptance.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from qwen_booster.cli import environment, local_model, prompt_tokens
from qwen_booster.runtime import HybridEngine


def measure(engine, tokens, generated_limit, *, reuse=True):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter()
    cache = engine.prepare(tokens, reuse=reuse)
    prepare_seconds = time.perf_counter() - started
    cached_tokens = engine.last_cached_tokens
    generated, first, last = [], None, None
    iterator = generate_step(mx.array(tokens[-1:], dtype=mx.uint32), engine.model,
                             max_tokens=generated_limit, prompt_cache=cache,
                             prefill_step_size=engine.prefill_step_size)
    try:
        for token, _ in iterator:
            now = time.perf_counter()
            first = now if first is None else first
            last = now
            generated.append(int(token))
    finally:
        iterator.close()
    committed = engine.commit(tokens + generated, cache)
    return {"prepare_seconds": prepare_seconds, "cached_tokens": cached_tokens,
            "processed_prompt_tokens": len(tokens) - 1 - cached_tokens,
            "generated": generated, "first_token_seconds": (first - started) if first else None,
            "decode_seconds": (last - first) if first and last else None,
            "commit_published": committed,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "active_mlx_bytes": mx.get_active_memory(),
            "prefix_stats": engine.prefixes.stats(),
            "disk_stats": engine.disk.stats() if engine.disk else None}


def engine_args(args):
    return dict(prefill_step_size=2048, cache_bytes=args.cache_mb * 1_000_000,
                active_memory_budget=args.memory_mb * 1_000_000,
                block_size=args.block_size, disk_root=args.disk_dir,
                disk_bytes=(args.cache_mb * 1_000_000) if args.disk_dir else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--length", type=int, default=8193)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--cache-mb", type=int, default=24000)
    parser.add_argument("--memory-mb", type=int, default=110000)
    parser.add_argument("--allocator-cache-mb", type=int, default=2000)
    parser.add_argument("--disk-dir", help="Directory for the disk restore case")
    parser.add_argument("--restore-only", action="store_true",
                        help="New-process disk restore against an existing --disk-dir")
    parser.add_argument("--output", default="results/hybrid-cache-native-v1.jsonl")
    args = parser.parse_args()
    if args.length < args.block_size + 2 or args.tokens < 1:
        parser.error("--length must exceed the block size by at least 2; --tokens must be positive")
    if args.restore_only and not args.disk_dir:
        parser.error("--restore-only requires --disk-dir")
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    if args.disk_dir and not args.restore_only and Path(args.disk_dir).exists():
        raise SystemExit(f"Refusing to reuse an existing disk cache: {args.disk_dir}")

    os.environ.setdefault("MLX_GDN_PACKED", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import mlx.core as mx
    from mlx_lm import load

    path, config = local_model(args.model)
    mx.set_memory_limit(args.memory_mb * 1_000_000)
    mx.set_cache_limit(args.allocator_cache_mb * 1_000_000)
    started = time.perf_counter()
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - started

    tokens = prompt_tokens(tokenizer, args.length)
    boundary = (len(tokens) - 1) - (len(tokens) - 1) % args.block_size
    replacement = prompt_tokens(tokenizer, args.length + 64)[-1]
    if replacement == tokens[-1]:
        replacement = (tokens[-1] + 1) % 248320
    diverged = list(tokens)
    diverged[-1] = replacement
    extended = tokens + prompt_tokens(tokenizer, args.length + 8)[-8:]

    engine = HybridEngine(model, **engine_args(args))
    engine.check_memory()
    rows, cases, parity = [], {}, {}

    if args.restore_only:
        if engine.prefixes.stats()["entries"] != 0:
            raise SystemExit("restore-only expects an empty in-memory cache")
        result = measure(engine, tokens, args.tokens)
        rows.append({"type": "measurement", "case": "disk-restore", "reuse": True,
                     "prompt_tokens": len(tokens), **result})
        checks = {"disk_restore_hit": result["cached_tokens"] == boundary}
        cases["disk-restore"] = result
        print(f"disk-restore: cached={result['cached_tokens']} "
              f"prepare={result['prepare_seconds']:.3f}s", flush=True)
    else:
        order = [("cold", engine, tokens, True),
                 ("warm", engine, tokens, True),
                 ("divergent-tail", engine, diverged, True),
                 ("extended-prefix", engine, extended, True),
                 ("no-reuse-control", engine, tokens, False)]
        for name, target, prompt, reuse in order:
            result = measure(target, prompt, args.tokens, reuse=reuse)
            rows.append({"type": "measurement", "case": name, "reuse": reuse,
                         "prompt_tokens": len(prompt), **result})
            cases[name] = result
            print(f"{name}: cached={result['cached_tokens']} "
                  f"prepare={result['prepare_seconds']:.3f}s "
                  f"generated={result['generated'][:4]}...", flush=True)
        if args.disk_dir:
            fresh = HybridEngine(model, **engine_args(args))
            result = measure(fresh, tokens, args.tokens)
            rows.append({"type": "measurement", "case": "disk-restore", "reuse": True,
                         "prompt_tokens": len(tokens), **result})
            cases["disk-restore"] = result
            print(f"disk-restore: cached={result['cached_tokens']} "
                  f"prepare={result['prepare_seconds']:.3f}s", flush=True)

        reference = cases["cold"]["generated"]
        for name in ("warm", "no-reuse-control", "disk-restore"):
            if name in cases:
                parity[name] = cases[name]["generated"] == reference
        checks = {
            "warm_hit_at_boundary": cases["warm"]["cached_tokens"] == boundary,
            "divergent_tail_hit_at_boundary": cases["divergent-tail"]["cached_tokens"] == boundary,
            "extended_prefix_hit_at_boundary": cases["extended-prefix"]["cached_tokens"] == boundary,
            "warm_prepare_faster_than_cold":
                cases["warm"]["prepare_seconds"] < cases["cold"]["prepare_seconds"],
            "control_is_cold": cases["no-reuse-control"]["cached_tokens"] == 0,
            "greedy_parity": all(parity.values()),
        }
        if args.disk_dir:
            checks["disk_restore_hit"] = cases["disk-restore"]["cached_tokens"] == boundary

    status = "complete" if all(checks.values()) else "failed"
    summary = {"status": status, "checks": checks, "boundary": boundary,
               "block_size": args.block_size, "load_seconds": load_seconds,
               "disk_dir": args.disk_dir, "restore_only": args.restore_only,
               "daily_use_validated": False,
               "scope": ("engine-level greedy checks on one synthetic token prompt; "
                         "not throughput, quality or daily-use acceptance")}
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = environment()
    metadata.update(model_path=str(path), model_config=config, args=vars(args),
                    token_sequence_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    scope=summary["scope"])
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["parity"] = parity
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if status != "complete":
        raise SystemExit("hybrid cache validation failed; keep the partial evidence")


if __name__ == "__main__":
    main()
