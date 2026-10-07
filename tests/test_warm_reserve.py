from types import SimpleNamespace

import pytest

from qwen_booster.warm_reserve import patch_warm_reserve


def setup_patch(allow_mtp=False):
    reports = []

    class Plan:
        def prepare(self, lengths, **kwargs):
            self.lengths = lengths
            return 120

        def reserve_bytes(self, live_bytes=0):
            return 120 - live_bytes

    class Coordinator:
        def __init__(self):
            self.model = 'flash'
            self.enabled = self.is_checkpoint = True
            self.calls = []
            self.manager = SimpleNamespace(memory_plan=Plan(), prepare_prefill=self.calls.append)

        def prepare_prefill(self, prompt_lengths, *, prefill_step_size=None, prefix_lengths=None):
            self.manager.memory_plan.prepare(prompt_lengths)
            self.manager.prepare_prefill(120)

        def merge_rows(self, picks, prefix_lens, *, kv_quant_config=None):
            self.prepare_prefill([200001] * len(picks), prefix_lengths=prefix_lens)
            return picks

    class Batch:
        def __init__(self, model, input_ids, apc_coordinator, warm_cache=None,
                     draft_model=None, draft_kind=None, kv_bits=None, right_pad_per_row=None,
                     fail=False):
            self.prompt_cache = warm_cache
            apc_coordinator.prepare_prefill([200001] * len(input_ids), prefix_lengths=[200000] * len(input_ids))
            if fail:
                raise RuntimeError('original error')

    patch_warm_reserve(Batch, Coordinator, accepts_model=lambda m: m == 'flash',
                       allow_mtp=allow_mtp, cache_bytes=lambda cache: cache, report=lambda *args: reports.append(args))
    return Batch, Coordinator(), reports


def test_both_post_restore_paths_credit_live_state_and_keep_copy_reserve():
    Batch, c, reports = setup_patch()
    picks = [{'warm_cache': 59}]
    assert c.merge_rows(picks, [200000]) is picks
    Batch('flash', [[1]], c, warm_cache=59)
    assert c.calls == [61, 61]
    assert [r[0] for r in reports] == ['merge', 'batch']
    c.prepare_prefill([200001])
    assert c.calls[-1] == 120  # Cold/pre-lookup admission receives no credit.


@pytest.mark.parametrize('override', [dict(draft_model=object()), dict(draft_kind='mtp'),
                                    dict(kv_bits=4), dict(right_pad_per_row=[1])])
def test_unsupported_batch_paths_keep_original_budget(override):
    Batch, c, reports = setup_patch()
    Batch('flash', [[1]], c, warm_cache=59, **override)
    assert c.calls == [120]
    assert not reports


def test_unknown_cache_model_and_multiple_rows_fail_closed():
    Batch, c, reports = setup_patch()
    Batch('flash', [[1]], c, warm_cache=None)
    Batch('other', [[1]], c, warm_cache=59)
    c.merge_rows([{'warm_cache': 59}, {'warm_cache': 59}], [200000, 200000])
    c.merge_rows([{'warm_cache': 59}], [200000], kv_quant_config={'bits': 4})
    c.merge_rows([{'warm_cache': None}], [200000])
    assert c.calls == [120] * 5
    assert not reports


def test_credit_capped_and_context_cleared_on_error():
    Batch, c, reports = setup_patch()
    with pytest.raises(RuntimeError, match='original error'):
        Batch('flash', [[1]], c, warm_cache=1000, fail=True)
    c.prepare_prefill([200001], prefix_lengths=[200000])
    assert c.calls == [60, 120]
    assert reports[0][2] == 60


def test_explicit_mtp_credit_keeps_other_drafts_excluded():
    Batch, c, reports = setup_patch(allow_mtp=True)
    Batch('flash', [[1]], c, warm_cache=59, draft_model=object(), draft_kind='mtp')
    Batch('flash', [[1]], c, warm_cache=59, draft_model=object(), draft_kind='other')
    Batch('flash', [[1]], c, warm_cache=59, draft_kind='mtp')
    assert c.calls == [61, 120, 120]
    assert len(reports) == 1
