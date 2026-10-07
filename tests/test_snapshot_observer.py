import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_snapshot_observer_preserves_results_errors_and_scope():
    spec = importlib.util.spec_from_file_location(
        'snapshot_observer', Path(__file__).parents[1] / 'scripts/trace_snapshot_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phases, calls = [], []
    result = object()

    def extract(value, *, fail=False):
        calls.append(value)
        if fail:
            raise ValueError('extraction failed')
        return result

    api = SimpleNamespace(extract_prompt_cache_from_batch=extract,
                          _clone_prompt_cache_for_apc=lambda value: value)

    class Coordinator:
        def store_checkpoint(self, value, *, fail=False):
            return api._clone_prompt_cache_for_apc(
                api.extract_prompt_cache_from_batch(value, fail=fail))

    module.install_snapshot_observer(Coordinator, api,
                                     lambda phase, coordinator: phases.append(phase))
    c = Coordinator()
    assert c.store_checkpoint(3) is result
    assert phases == ['checkpoint.before', 'extract.before', 'extract.after',
                      'clone.before', 'clone.after', 'checkpoint.after']
    with pytest.raises(ValueError, match='extraction failed'):
        c.store_checkpoint(4, fail=True)
    assert phases[-4:] == ['checkpoint.before', 'extract.before',
                           'extract.after', 'checkpoint.after']
    count = len(phases)
    assert api.extract_prompt_cache_from_batch(5) is result
    assert len(phases) == count
    assert calls == [3, 4, 5]
