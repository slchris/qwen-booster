"""Opt-in resident MLP weights for large prefill matrices; decode stays quantized."""

import time


def expand_prefill_weights(model, memory_budget_bytes):
    import mlx.core as mx
    import mlx.nn as nn

    projections = []
    estimated_bytes = 0
    for layer in model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            original = getattr(layer.mlp, name)
            if (not isinstance(original, nn.QuantizedLinear) or "bias" in original
                    or original.mode != "affine" or original.bits != 4):
                raise ValueError("expanded prefill requires bias-free affine Q4 MLP projections")
            estimated_bytes += original.weight.shape[0] * (
                original.weight.shape[1] * 32 // original.bits) * original.scales.itemsize
            projections.append((layer.mlp, name, original))
    mx.eval(model.parameters())
    active_before = mx.get_active_memory()
    if active_before + estimated_bytes >= memory_budget_bytes:
        raise ValueError("expanded prefill weights exceed the active-memory budget before KV/workspace; "
                         "increase --memory-mb or omit --expanded-prefill")

    class ExpandedPrefillLinear(nn.QuantizedLinear):
        def __init__(self, original):
            # Preserve QuantizedLinear identity and flat parameters: speculative
            # verification dispatches specialized exact kernels by isinstance.
            nn.Module.__init__(self)
            self.weight, self.scales, self.biases = original.weight, original.scales, original.biases
            self.bits, self.group_size, self.mode = original.bits, original.group_size, original.mode
            self.expanded = mx.dequantize(original.weight, original.scales, original.biases,
                group_size=original.group_size, bits=original.bits, mode=original.mode)
            mx.eval(self.expanded)

        def __call__(self, x):
            # The single-token QMM/GEMV path has different arithmetic. Preserve it,
            # including small final prefill chunks and speculative verification.
            if x.shape[-2] >= 128:
                return x @ self.expanded.T
            return super().__call__(x)

    start = time.perf_counter()
    for owner, name, original in projections:
        setattr(owner, name, ExpandedPrefillLinear(original))
    mx.synchronize()
    return {"enabled": True, "minimum_matrix_tokens": 128,
            "additional_resident_bytes": mx.get_active_memory() - active_before,
            "setup_seconds": time.perf_counter() - start,
            "decode_uses_original_quantized_weights": True}
