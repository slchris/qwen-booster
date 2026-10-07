import json

import pytest

from qwen_booster.models import identify_model, resolve_local_model


@pytest.mark.parametrize('kind,layers,hidden,experts,expected', [
    ('qwen3_5', 64, 5120, 0, 'qwen3.8-27b'),
    ('qwen4_exp', 48, 2560, 512, 'qwen3.8-flash-next'),
])
def test_models_require_full_weights_before_serving(tmp_path, kind, layers, hidden, experts, expected):
    config = {'model_type': kind, 'text_config': {'num_hidden_layers': layers,
              'hidden_size': hidden, 'num_experts': experts}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'weights.safetensors'}}))
    assert resolve_local_model(tmp_path, require_weights=False)[2].id == expected
    with pytest.raises(ValueError, match='Missing model shards'):
        resolve_local_model(tmp_path)
    (tmp_path / 'weights.safetensors').write_bytes(b'fixture')
    assert resolve_local_model(tmp_path)[2].id == expected


def test_flash_cannot_use_dense_expansion_and_unknown_architecture_rejected():
    profile = identify_model({'model_type': 'qwen4_exp', 'text_config': {
        'num_hidden_layers': 48, 'hidden_size': 2560, 'num_experts': 512}})
    assert not profile.dense_prefill_expansion
    assert profile.draft_type == 'qwen4_exp_mtp'
    with pytest.raises(ValueError, match='Unsupported model'):
        identify_model({'model_type': 'qwen4_exp', 'num_hidden_layers': 48,
                        'hidden_size': 2560, 'num_experts': 128})
