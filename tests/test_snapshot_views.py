from types import SimpleNamespace

import pytest

from qwen_booster.snapshot_views import patch_snapshot_views


def test_view_scope_keeps_original_store_and_falls_back_safely():
    calls = []
    api = SimpleNamespace(extract_prompt_cache_from_batch=lambda rows, idx: ('original', rows, idx))

    class Coordinator:
        enabled = is_checkpoint = True
        model = 'flash'

        def store_checkpoint(self, rows, *, fail=False):
            calls.append('admission')
            result = api.extract_prompt_cache_from_batch(rows, 0)
            if fail:
                raise ValueError('original failure')
            calls.append('detached clone')
            return result

    patch_snapshot_views(Coordinator, api, accepts_model=lambda m: m == 'flash',
                         borrow=lambda rows, idx: ('views', rows) if rows == 'supported' else None)
    c = Coordinator()
    assert c.store_checkpoint('supported') == ('views', 'supported')
    assert calls == ['admission', 'detached clone']
    assert c.store_checkpoint('unknown') == ('original', 'unknown', 0)
    c.model = 'other'
    assert c.store_checkpoint('supported') == ('original', 'supported', 0)
    c.model = 'flash'
    with pytest.raises(ValueError, match='original failure'):
        c.store_checkpoint('supported', fail=True)
    assert api.extract_prompt_cache_from_batch('supported', 0) == ('original', 'supported', 0)
    c.enabled = False
    assert c.store_checkpoint('supported') == ('original', 'supported', 0)
