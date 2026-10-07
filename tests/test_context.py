import pytest

from qwen_booster.context import ContextBudget, full_attention_kv_bytes


def test_output_reservation_and_native_boundary():
    budget = ContextBudget.from_config({"text_config": {"max_position_embeddings": 262144}}, 262144)
    budget.admit(258048, 4096)
    with pytest.raises(ValueError, match="not truncated"):
        budget.admit(258049, 4096)
    with pytest.raises(ValueError, match="extended context"):
        ContextBudget.from_config({"max_position_embeddings": 262144}, 1000000)


def test_hybrid_kv_estimate_excludes_recurrent_layers():
    config = {"layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
              "num_key_value_heads": 4, "head_dim": 256}
    assert full_attention_kv_bytes(config, 262144) == 16 * 1024**3


@pytest.mark.parametrize("value", [0, -1, True, None])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        ContextBudget.from_config({"max_position_embeddings": 262144}, value)
