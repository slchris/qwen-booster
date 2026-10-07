"""Screen native Q4 GEMM versus expanded-weight GEMM on actual model MLP weights.

No runtime mutation. Reports expansion cost/memory and numerical error; a faster
microbenchmark does not authorize enabling this path without full-model checks.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from qwen_booster.cli import environment, local_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 128, 512, 2048])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite results")
    if any(n < 1 for n in args.lengths):
        parser.error("lengths must be positive")
    path, _ = local_model(args.model)
    mx.set_cache_limit(2_000_000_000)
    model, _ = load(path)
    mx.eval(model.parameters())
    rows = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(model.layers[0].mlp, name)
        if not isinstance(layer, nn.QuantizedLinear) or "bias" in layer:
            raise ValueError("expected bias-free quantized MLP projection")
        mx.synchronize()
        start = time.perf_counter()
        weight = mx.dequantize(layer.weight, layer.scales, layer.biases,
                              group_size=layer.group_size, bits=layer.bits, mode=layer.mode)
        mx.eval(weight)
        expansion_seconds = time.perf_counter() - start
        for length in args.lengths:
            mx.random.seed(1729)
            x = mx.random.normal((1, length, weight.shape[1])).astype(weight.dtype)
            mx.eval(x)
            reference = layer(x)
            candidate = x @ weight.T
            mx.eval(reference, candidate)
            delta = candidate.astype(mx.float32) - reference.astype(mx.float32)
            error = {"max_abs": mx.max(mx.abs(delta)).item(),
                     "relative_l2": (mx.linalg.norm(delta) /
                                     mx.maximum(mx.linalg.norm(reference.astype(mx.float32)), 1e-12)).item()}
            samples = {"native": [], "expanded": []}
            for repeat in range(6):
                for mode in (["native", "expanded"] if repeat % 2 == 0 else ["expanded", "native"]):
                    mx.synchronize()
                    start = time.perf_counter()
                    result = layer(x) if mode == "native" else x @ weight.T
                    mx.eval(result)
                    elapsed = time.perf_counter() - start
                    if repeat:
                        samples[mode].append(elapsed)
            row = {"projection": name, "tokens": length, "dtype": str(weight.dtype),
                   "expanded_weight_bytes": weight.nbytes, "expansion_seconds": expansion_seconds,
                   "samples_seconds": samples, "error": error,
                   "micro_speedup": statistics.median(samples["native"]) / statistics.median(samples["expanded"])}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples_seconds"}), flush=True)
        del weight, reference, candidate, result, x, delta
        mx.clear_cache()
    output.write_text(json.dumps({"environment": environment(), "scope": __doc__,
                                 "rows": rows, "enabled_in_runtime": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
