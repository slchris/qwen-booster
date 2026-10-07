"""Paired full-model screen: expanded MLP weights for prefill, original Q4 for decode.

Both modes keep the same expanded weights resident for timing fairness. Report the
additional resident bytes and one-time expansion cost separately. No runtime patch.
"""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or args.repeats < 1 or args.tokens < 2:
        parser.error("require new output, positive repeats and at least two output tokens")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from qwen_booster.cli import environment, local_model, measure, prompt_tokens
    from qwen_booster.context import ContextBudget
    from qwen_booster.prefill_weights import expand_prefill_weights

    path, config = local_model(args.model)
    budget = ContextBudget.from_config(config, config.get("text_config", config)["max_position_embeddings"])
    for length in args.lengths:
        if length < 129:
            parser.error("screen needs at least 129 prompt tokens (last token is decoded separately)")
        budget.admit(length, args.tokens)
    mx.set_memory_limit(64_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    enabled = False
    calls = 0
    setup = expand_prefill_weights(model, 64_000_000_000)
    projection_type = type(model.layers[0].mlp.gate_proj)
    runtime_call = projection_type.__call__

    def selected_call(self, x):
        nonlocal calls
        if enabled:
            if x.shape[-2] >= 128:
                calls += 1
            return runtime_call(self, x)
        return nn.QuantizedLinear.__call__(self, x)

    # Use the actual service implementation; toggle only inside this benchmark.
    projection_type.__call__ = selected_call
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "environment": environment(), "scope": __doc__,
            "additional_resident_bytes": setup["additional_resident_bytes"],
            "setup_seconds": setup["setup_seconds"], "implementation": "qwen_booster.prefill_weights",
            "default_enabled_in_runtime": False, "prefill_min_tokens": 128}) + "\n")
        handle.flush()
        for length in args.lengths:
            tokens = prompt_tokens(tokenizer, length)
            for repeat in range(args.repeats + 1):
                results = {}
                for mode in (["native", "expanded"] if repeat % 2 == 0 else ["expanded", "native"]):
                    enabled = mode == "expanded"
                    calls = 0
                    row = measure(model, tokens, 2048, args.tokens)
                    row.update(type="measurement", mode=mode, repeat=repeat,
                               warmup=repeat == 0, expanded_calls=calls)
                    if enabled and not calls:
                        raise RuntimeError("expanded prefill path was not executed")
                    results[mode] = row
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                parity = results["native"]["output_token_ids"] == results["expanded"]["output_token_ids"]
                pair = {"type": "pair", "prompt_tokens": length, "repeat": repeat,
                    "warmup": repeat == 0, "token_parity": parity,
                    "ttft_speedup": results["native"]["ttft_seconds"] / results["expanded"]["ttft_seconds"]
                        if parity else None}
                handle.write(json.dumps(pair) + "\n")
                handle.flush()
                print(json.dumps(pair), flush=True)
                if repeat:
                    pairs.append(pair)
    summary = []
    for length in args.lengths:
        selected = [p for p in pairs if p["prompt_tokens"] == length]
        parity = all(p["token_parity"] for p in selected)
        summary.append({"prompt_tokens": length, "token_parity": parity,
            "median_paired_ttft_speedup": statistics.median(p["ttft_speedup"] for p in selected)
                if parity else None})
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not all(p["token_parity"] for p in pairs):
        raise SystemExit("Output parity failed; keep expanded weights disabled")


if __name__ == "__main__":
    main()
