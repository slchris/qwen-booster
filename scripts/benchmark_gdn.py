"""Compare upstream Metal GDN layouts at Qwen3.8-27B head dimensions."""

import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import (
    gated_delta_kernel, gated_delta_kernel_unpacked, gated_delta_ops, normalize_qk,
)
from qwen_booster.cli import environment


def main():
    output = Path("results/gdn-layout.json")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    mx.random.seed(42)
    rows = []
    for length in (1, 8, 128, 512, 2048):
        q = mx.random.normal((1, length, 16, 128)).astype(mx.bfloat16)
        k = mx.random.normal(q.shape).astype(mx.bfloat16)
        q, k = normalize_qk(q, k, inv_scale=128**-0.5, eps=1e-6)
        v = mx.random.normal((1, length, 48, 128)).astype(mx.bfloat16)
        g = mx.exp(-mx.random.uniform(shape=(1, length, 48)))
        beta = mx.sigmoid(mx.random.normal(g.shape)).astype(mx.bfloat16)
        state = mx.random.normal((1, 48, 128, 128)) * 0.01
        values = (q, k, v, g, beta, state)
        mx.eval(values)
        expected = gated_delta_kernel(*values)
        actual = gated_delta_kernel_unpacked(*values)
        mx.eval(expected, actual)
        equal = all(mx.array_equal(a, b).item() for a, b in zip(expected, actual))
        reference_error = None
        if length <= 8:
            reference = gated_delta_ops(*values)
            mx.eval(reference)
            reference_error = [mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
                               for a, b in zip(expected, reference)]
        times = {"packed": [], "unpacked": []}
        for repeat in range(11):
            variants = [("packed", gated_delta_kernel), ("unpacked", gated_delta_kernel_unpacked)]
            if repeat % 2:
                variants.reverse()
            for name, fn in variants:
                mx.synchronize()
                start = time.perf_counter()
                mx.eval(fn(*values))
                elapsed = time.perf_counter() - start
                if repeat:
                    times[name].append(elapsed)
        row = {"tokens": length, "bitwise_equal": equal, "ops_reference_max_abs_error": reference_error,
               "median_seconds": {name: statistics.median(t) for name, t in times.items()}, "samples": times}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"environment": environment(), "rows": rows,
        "scope": "Isolated upstream kernels on synthetic normalized bf16 inputs; not end-to-end model speed"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
