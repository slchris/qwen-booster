import importlib.util
from pathlib import Path

import pytest


def test_observer_preserves_calls_and_clears_context_after_error():
    spec = importlib.util.spec_from_file_location('warm_observer', Path(__file__).parents[1] / 'scripts/trace_warm_reserve_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls, observations = [], []

    class Coordinator:
        def prepare_prefill(self, lengths, *, prefix_lengths=None):
            calls.append((lengths, prefix_lengths))
            return 'unchanged'

    class Batch:
        def __init__(self, coordinator, warm_cache=None, fail=False):
            self._apc_coordinator = coordinator
            self.prompt_cache = warm_cache
            assert coordinator.prepare_prefill([200001], prefix_lengths=[200000]) == 'unchanged'
            if fail:
                raise RuntimeError('original failure')

    module.install_observer(Batch, Coordinator, lambda phase, c, b: observations.append(phase))
    c = Coordinator()
    cache = [object()]
    assert Batch(c, cache).prompt_cache is cache
    assert observations == ['before', 'after']
    Batch(c)
    assert observations == ['before', 'after']
    with pytest.raises(RuntimeError, match='original failure'):
        Batch(c, cache, fail=True)
    c.prepare_prefill([1])
    assert observations == ['before', 'after', 'before', 'after']
    assert calls == [([200001], [200000])] * 3 + [([1], None)]
