"""Screen default versus forced-fused SDPA for Qwen's actual attention shapes.

Synthetic tensors isolate the kernel. This is not full-model quality evidence or
an end-to-end speedup claim. Run with other GPU inference workloads stopped.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kv-lengths", type=int, nargs="+", default=[2048, 8192, 32768, 65536])
    parser.add_argument("--query-length", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--query-slice", type=int, default=0,
                        help="Also screen unfused attention with query-only slicing; 0 disables")
    args = parser.parse_args()
    if args.query_length < 1 or args.repeats < 1 or any(n < args.query_length for n in args.kv_lengths):
        parser.error("require positive repeats and 1 <= query length <= each KV length")
    if args.query_slice and not 8 < args.query_slice < args.query_length:
        parser.error("query-slice must be >8 and smaller than query length, or zero")
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite results")

    import mlx.core as mx
    from qwen_booster.cli import environment, local_model

    _, config = local_model(None)
    config = config.get("text_config", config)
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    dim = config["head_dim"]
    dtype = getattr(mx, config.get("torch_dtype", config.get("dtype", "bfloat16")))
    mx.set_memory_limit(64_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "environment": environment(),
            "scope": __doc__, "dtype": str(dtype), "heads": heads, "kv_heads": kv_heads,
            "head_dim": dim, "enabled_in_runtime": False,
            "source": "https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp"}) + "\n")
        handle.flush()
        for length in args.kv_lengths:
            mx.random.seed(1729)
            q = mx.random.normal((1, heads, args.query_length, dim)).astype(dtype)
            k = mx.random.normal((1, kv_heads, length, dim)).astype(dtype)
            v = mx.random.normal((1, kv_heads, length, dim)).astype(dtype)
            mx.eval(q, k, v)

            def call(mode):
                if mode == "query_sliced":
                    parts = []
                    key_positions = mx.arange(length)[None, :]
                    for start in range(0, args.query_length, args.query_slice):
                        stop = min(start + args.query_slice, args.query_length)
                        # Retain all K/V columns, including masked future columns,
                        # to preserve the original softmax reduction dimensions.
                        positions = mx.arange(length - args.query_length + start,
                                              length - args.query_length + stop)[:, None]
                        part = mx.fast.scaled_dot_product_attention(
                            q[:, :, start:stop, :], k, v, scale=dim ** -0.5,
                            mask=positions >= key_positions, force_fused=False)
                        mx.eval(part)
                        parts.append(part)
                    return mx.concatenate(parts, axis=2)
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=dim ** -0.5,
                    mask="causal", force_fused=mode == "forced_fused")

            reference = call("default")
            candidate = call("forced_fused")
            mx.eval(reference, candidate)
            delta = candidate.astype(mx.float32) - reference.astype(mx.float32)
            error = {"max_abs": mx.max(mx.abs(delta)).item(),
                     "relative_l2": (mx.linalg.norm(delta) / mx.maximum(
                         mx.linalg.norm(reference.astype(mx.float32)), 1e-12)).item(),
                     "finite": bool(mx.all(mx.isfinite(candidate)).item())}
            sliced_error = None
            if args.query_slice:
                sliced = call("query_sliced")
                mx.eval(sliced)
                diff = sliced.astype(mx.float32) - reference.astype(mx.float32)
                sliced_error = {"max_abs": mx.max(mx.abs(diff)).item(),
                                "bitwise_equal": bool(mx.array_equal(sliced, reference).item()),
                                "finite": bool(mx.all(mx.isfinite(sliced)).item())}
                del sliced, diff
            del reference, candidate, delta
            samples = {"default": [], "forced_fused": []}
            if args.query_slice:
                samples["query_sliced"] = []
            for repeat in range(args.repeats + 1):
                modes = list(samples)
                order = modes if repeat % 2 == 0 else modes[::-1]
                for mode in order:
                    mx.synchronize()
                    mx.clear_cache()
                    mx.reset_peak_memory()
                    active_before = mx.get_active_memory()
                    start = time.perf_counter()
                    result = call(mode)
                    mx.eval(result)
                    elapsed = time.perf_counter() - start
                    if repeat:
                        samples[mode].append({
                            "seconds": elapsed, "peak_bytes": mx.get_peak_memory(),
                            "active_before_bytes": active_before})
                    del result
            row = {"type": "measurement", "query_tokens": args.query_length, "kv_tokens": length,
                   "error": error, "samples": samples,
                   "query_slice": args.query_slice, "query_sliced_error": sliced_error,
                   "query_sliced_micro_speedup": (
                       statistics.median(s["seconds"] for s in samples["default"]) /
                       statistics.median(s["seconds"] for s in samples["query_sliced"])
                       if args.query_slice else None),
                   "micro_speedup": statistics.median(s["seconds"] for s in samples["default"]) /
                                    statistics.median(s["seconds"] for s in samples["forced_fused"])}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
            del q, k, v
            mx.clear_cache()


if __name__ == "__main__":
    main()
