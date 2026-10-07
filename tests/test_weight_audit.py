import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


spec = importlib.util.spec_from_file_location('audit_model_weights',
    Path(__file__).parents[1] / 'scripts' / 'audit_model_weights.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_index_requires_present_valid_and_correctly_assigned_tensors(tmp_path):
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'a': 'shard.safetensors'}}))
    assert not audit.audit_weights(tmp_path)['complete']
    shard = tmp_path / 'shard.safetensors'
    save_file({'b': np.zeros((2, 3), dtype=np.float32)}, shard)
    assert not audit.audit_weights(tmp_path)['complete']
    save_file({'a': np.zeros((2, 3), dtype=np.float32)}, shard)
    assert audit.audit_weights(tmp_path)['complete']
    manifest = {'weight_files': [{'name': shard.name, 'size': shard.stat().st_size + 1}]}
    assert not audit.audit_weights(tmp_path, manifest)['complete']
    shard.write_bytes(shard.read_bytes()[:-1])
    assert not audit.audit_weights(tmp_path)['complete']


def test_untrusted_index_cannot_read_outside_model_directory(tmp_path):
    (tmp_path / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'a': '../other.safetensors'}}))
    with pytest.raises(ValueError, match='escapes'):
        audit.audit_weights(tmp_path)


def test_snapshot_blob_symlinks_are_supported(tmp_path):
    blob = tmp_path / 'blob'
    save_file({'a': np.zeros((1,), dtype=np.float32)}, blob)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'shard.safetensors').symlink_to(blob)
    (snapshot / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'a': 'shard.safetensors'}}))
    assert audit.audit_weights(snapshot)['complete']
