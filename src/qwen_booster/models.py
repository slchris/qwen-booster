"""Local model identification and capability boundaries for workload comparisons."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    id: str
    model_type: str
    layers: int
    hidden_size: int
    experts: int
    draft_type: str
    dense_prefill_expansion: bool


PROFILES = (
    ModelProfile('qwen3.8-27b', 'qwen3_5', 64, 5120, 0, 'qwen3_5_mtp', True),
    ModelProfile('qwen3.8-flash-next', 'qwen4_exp', 48, 2560, 512, 'qwen4_exp_mtp', False),
)


def identify_model(config):
    text = config.get('text_config', config)
    for profile in PROFILES:
        if (config.get('model_type') == profile.model_type
                and text.get('num_hidden_layers') == profile.layers
                and text.get('hidden_size') == profile.hidden_size
                and text.get('num_experts', 0) == profile.experts):
            return profile
    raise ValueError('Unsupported model configuration; add and validate a model profile first')


def resolve_local_model(value, *, require_weights=True):
    path = Path(value).expanduser().resolve()
    config = json.loads((path / 'config.json').read_text())
    profile = identify_model(config)
    if require_weights:
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Model index has no weight map')
        missing = [name for name in sorted(set(weight_map.values())) if not (path / name).is_file()]
        if missing:
            raise ValueError(f'Missing model shards: {missing}')
    return path, config, profile
