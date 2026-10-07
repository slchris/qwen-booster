"""GPU regression for budget rejection and upstream exact MTP dispatch identity.

Run in the pinned VLM environment with PYTHONPATH=src. Tiny synthetic projections
check the integration invariant; full-model quality and speed are separate gates.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite results")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.speculative.ops.linear import _target_verify_linear, _use_target_verify_dense
    from qwen_booster.prefill_weights import expand_prefill_weights

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=4)
            self.up_proj = nn.QuantizedLinear(128, 256, bias=False, group_size=64, bits=4)
            self.down_proj = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    mx.random.seed(1729)
    model = Model()
    names = ("gate_proj", "up_proj", "down_proj")
    originals = {name: getattr(model.layers[0].mlp, name) for name in names}
    try:
        expand_prefill_weights(model, 1)
        raise AssertionError("insufficient budget was not rejected")
    except ValueError as error:
        assert "budget" in str(error)
    assert all(getattr(model.layers[0].mlp, name) is original for name, original in originals.items())
    report = expand_prefill_weights(model, 1_000_000_000)
    rows = []
    for name, original in originals.items():
        candidate = getattr(model.layers[0].mlp, name)
        assert isinstance(candidate, nn.QuantizedLinear)
        for length in (1, 3):
            x = mx.random.normal((1, length, original.weight.shape[1] * 32 // original.bits)).astype(mx.bfloat16)
            reference = _target_verify_linear(original, x)
            actual = _target_verify_linear(candidate, x)
            mx.eval(reference, actual)
            error = mx.max(mx.abs(actual.astype(mx.float32) - reference.astype(mx.float32))).item()
            original_dispatch = _use_target_verify_dense(original, x)
            candidate_dispatch = _use_target_verify_dense(candidate, x)
            assert original_dispatch == candidate_dispatch
            if length == 3:
                assert original_dispatch, "this regression requires the exact verification kernel"
            assert error == 0, (name, length, error)
            rows.append({"projection": name, "tokens": length, "max_abs_error": error,
                         "exact_dispatch_preserved": True,
                         "exact_dispatch_selected": candidate_dispatch})
    output.write_text(json.dumps({"scope": __doc__, "budget_rejection_before_mutation": True,
        "setup": report, "checks": rows, "all_passed": True}, indent=2) + "\n")
    print("Budget and exact MTP dispatch regression passed", flush=True)


if __name__ == "__main__":
    main()
