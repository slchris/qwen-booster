"""Experimental packed gate/up projection using existing MLX Metal QMM.

Quantized rows are concatenated without requantization. This is a scheduling and
layout experiment, not a new quantization algorithm or custom Metal kernel.
"""


def fused_mlp(original):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.activations import swiglu

    gate, up = original.gate_proj, original.up_proj
    if not all(isinstance(layer, nn.QuantizedLinear) for layer in (gate, up)):
        raise ValueError("gate/up fusion requires quantized linear layers")
    if any(getattr(gate, key) != getattr(up, key) for key in ("bits", "group_size", "mode")):
        raise ValueError("gate/up quantization settings differ")
    if "bias" in gate or "bias" in up or gate.mode != "affine":
        raise ValueError("initial fusion supports bias-free affine quantization only")

    class FusedMLP(nn.Module):
        def __init__(self, gate, up, down):
            super().__init__()
            self.weight = mx.concatenate([gate.weight, up.weight], axis=0)
            self.scales = mx.concatenate([gate.scales, up.scales], axis=0)
            self.biases = mx.concatenate([gate.biases, up.biases], axis=0)
            self.bits, self.group_size, self.mode = gate.bits, gate.group_size, gate.mode
            self.split = gate.weight.shape[0]
            self.down_proj = down
            mx.eval(self.weight, self.scales, self.biases)

        def __call__(self, x):
            both = mx.quantized_matmul(x, self.weight, scales=self.scales,
                      biases=self.biases, transpose=True, group_size=self.group_size,
                      bits=self.bits, mode=self.mode)
            g, u = mx.split(both, [self.split], axis=-1)
            return self.down_proj(swiglu(g, u))

    return FusedMLP(gate, up, original.down_proj)


def apply_fusion(model):
    for layer in model.layers:
        layer.mlp = fused_mlp(layer.mlp)


def validate_fusion(args):
    import gc
    import json
    import statistics
    from pathlib import Path
    import mlx.core as mx
    from mlx_lm import load
    from .cli import environment, local_model, measure, prompt_tokens

    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    path, _ = local_model(args.model)
    model, tokenizer = load(path)
    original = model.layers[0].mlp
    candidate = fused_mlp(original)
    mx.random.seed(42)
    numeric = []
    for length in (1, 16, 512):
        x = mx.random.normal((1, length, 5120)).astype(mx.bfloat16)
        expected = original(x).astype(mx.float32)
        actual = candidate(x).astype(mx.float32)
        mx.eval(expected, actual)
        relative = (mx.linalg.norm(actual - expected) / mx.maximum(mx.linalg.norm(expected), 1e-8)).item()
        numeric.append({"tokens": length, "relative_l2_error": relative,
                        "max_abs_error": mx.max(mx.abs(actual - expected)).item()})
    del candidate, original
    gc.collect()
    # Sequential before/after is only a screening experiment; promotion requires
    # alternating fresh-process measurements to control temporal drift.
    tokens = prompt_tokens(tokenizer, 512)
    rows = []
    for mode in ("native", "fused"):
        if mode == "fused":
            apply_fusion(model)
        measure(model, tokens, 2048, 4)
        for repeat in range(3):
            row = measure(model, tokens, 2048, 64)
            row.update(mode=mode, repeat=repeat)
            rows.append(row)
            print(f"{mode} repeat={repeat}: prefill={row['prefill_tps']:.1f}, decode={row['decode_tps']:.1f}", flush=True)
    reference = rows[0]["output_token_ids"]
    parity = all(r["output_token_ids"] == reference for r in rows)
    ratios = {metric: statistics.median(r[metric] for r in rows if r["mode"] == "fused") /
                      statistics.median(r[metric] for r in rows if r["mode"] == "native")
              for metric in ("prefill_tps", "decode_tps")}
    report = {"environment": environment(), "numeric": numeric, "token_parity": parity,
              "ratios": ratios, "rows": rows, "provisional": True,
              "scope": "512 raw prompt tokens; 64 greedy output tokens; before/after screening only"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"token_parity": parity, "ratios": ratios, "numeric": numeric}, indent=2))
