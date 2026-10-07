"""Paired full-model screen of prefill attention candidates, isolated to this process.

Uses unchanged target weights, fresh caches, alternating run order and exact greedy
token comparison. Synthetic prompts and fixed output length (ignoring EOS) isolate
performance; separate task and HTTP checks are required before enabling a default.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-pairs", type=int, choices=[0, 1], default=1)
    parser.add_argument("--head-group-min-kv", type=int, default=0)
    parser.add_argument("--candidate", choices=["fused", "query-sliced", "head-group"], default="fused")
    parser.add_argument("--query-slice", type=int, default=512)
    args = parser.parse_args()
    if args.tokens < 2 or args.step < 1 or args.repeats < 1 or any(n < 2 for n in args.lengths):
        parser.error("invalid benchmark dimensions")
    if args.head_group_min_kv < 0:
        parser.error("head-group-min-kv must be nonnegative")
    if args.candidate == "query-sliced" and not 8 < args.query_slice < args.step:
        parser.error("query-slice must be >8 and smaller than prefill step")
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite results")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import qwen3_next
    from qwen_booster.cli import environment, local_model, measure, prompt_tokens
    from qwen_booster.context import ContextBudget

    path, config = local_model(args.model)
    native_limit = config.get("text_config", config)["max_position_embeddings"]
    budget = ContextBudget.from_config(config, native_limit)
    for length in args.lengths:
        budget.admit(length, args.tokens)
    mx.set_memory_limit(64_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    original = qwen3_next.scaled_dot_product_attention
    fused_calls = 0

    def fused(queries, keys, values, cache, scale, mask, sinks=None):
        nonlocal fused_calls
        if args.candidate == "head-group":
            if (queries.ndim == keys.ndim == values.ndim == 4
                    and queries.shape[0] == keys.shape[0] == values.shape[0] == 1
                    and queries.shape[1] == 24 and keys.shape[1] == values.shape[1] == 4
                    and queries.shape[-1] == keys.shape[-1] == values.shape[-1] == 256
                    and queries.shape[2] > 8 and keys.shape[2] > queries.shape[2]
                    and keys.shape[2] >= args.head_group_min_kv
                    and isinstance(mask, str) and mask == "causal"
                    and sinks is None and not hasattr(cache, "bits")):
                fused_calls += 1
                parts = []
                for head in range(4):
                    part = original(queries[:, head * 6:(head + 1) * 6],
                                    keys[:, head:head + 1], values[:, head:head + 1],
                                    cache, scale, mask, sinks)
                    mx.eval(part)
                    parts.append(part)
                return mx.concatenate(parts, axis=1)
            return original(queries, keys, values, cache, scale, mask, sinks)
        if args.candidate == "query-sliced":
            qlen, klen = queries.shape[2], keys.shape[2]
            if (qlen > args.query_slice and qlen % args.query_slice == 0
                    and klen > qlen and queries.shape[-1] == 256
                    and isinstance(mask, str) and mask == "causal"
                    and sinks is None and not hasattr(cache, "bits")):
                fused_calls += 1
                parts = []
                columns = mx.arange(klen)[None, :]
                for start in range(0, qlen, args.query_slice):
                    stop = start + args.query_slice
                    positions = mx.arange(klen - qlen + start, klen - qlen + stop)[:, None]
                    part = mx.fast.scaled_dot_product_attention(
                        queries[:, :, start:stop, :], keys, values, scale=scale,
                        mask=positions >= columns, force_fused=False)
                    mx.eval(part)
                    parts.append(part)
                return mx.concatenate(parts, axis=2)
            return original(queries, keys, values, cache, scale, mask, sinks)
        if queries.shape[2] > 8 and queries.shape[-1] == 256 and not hasattr(cache, "bits"):
            fused_calls += 1
            return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale,
                                                       mask=mask, sinks=sinks, force_fused=True)
        return original(queries, keys, values, cache, scale, mask, sinks)

    output.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "environment": environment(), "scope": __doc__,
            "target_config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), Path(qwen3_next.__file__),
                                        Path("src/qwen_booster/cli.py")]},
            "enabled_in_runtime": False, "candidate": args.candidate,
            "warmup_pairs": args.warmup_pairs, "formal_pairs_per_length": args.repeats,
            "head_group_min_kv": args.head_group_min_kv,
            "query_slice": args.query_slice if args.candidate == "query-sliced" else None}) + "\n")
        handle.flush()
        try:
            for length in args.lengths:
                tokens = prompt_tokens(tokenizer, length)
                for repeat in range(args.repeats + args.warmup_pairs):
                    warmup = repeat < args.warmup_pairs
                    results = {}
                    for mode in (["default", args.candidate] if repeat % 2 == 0 else [args.candidate, "default"]):
                        qwen3_next.scaled_dot_product_attention = fused if mode != "default" else original
                        fused_calls = 0
                        print(json.dumps({"stage": "measurement_started", "mode": mode,
                                          "prompt_tokens": length, "repeat": repeat,
                                          "warmup": warmup}), flush=True)
                        result = measure(model, tokens, args.step, args.tokens)
                        result.update(type="measurement", mode=mode, repeat=repeat,
                                      warmup=warmup, candidate_calls=fused_calls)
                        results[mode] = result
                        handle.write(json.dumps(result) + "\n")
                        handle.flush()
                        should_call = args.step > 8 and (args.candidate == "fused" or length > 2 * args.step)
                        if args.candidate == "head-group":
                            should_call = should_call and length > args.head_group_min_kv
                        if mode != "default" and should_call and fused_calls == 0:
                            raise RuntimeError("candidate did not reach the expected attention call path")
                    equal = results["default"]["output_token_ids"] == results[args.candidate]["output_token_ids"]
                    pair = {"type": "pair", "prompt_tokens": length, "repeat": repeat,
                            "warmup": warmup, "token_parity": equal,
                            "ttft_speedup": results["default"]["ttft_seconds"] / results[args.candidate]["ttft_seconds"]
                                if equal else None}
                    handle.write(json.dumps(pair) + "\n")
                    handle.flush()
                    print(json.dumps(pair), flush=True)
                    if not warmup:
                        pairs.append(pair)
        finally:
            qwen3_next.scaled_dot_product_attention = original
    summary = []
    for length in args.lengths:
        selected = [p for p in pairs if p["prompt_tokens"] == length]
        parity = bool(selected) and all(p["token_parity"] for p in selected)
        summary.append({"prompt_tokens": length, "token_parity": parity,
                        "median_paired_ttft_speedup": statistics.median(p["ttft_speedup"] for p in selected)
                            if parity else None})
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not all(p["token_parity"] for p in pairs):
        raise RuntimeError("candidate changes greedy outputs; keep the runtime default unchanged")


if __name__ == "__main__":
    main()
