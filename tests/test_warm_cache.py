from qwen_booster.warm_cache import patch_prompt_batch


def test_single_warm_hook_preserves_other_constructor_inputs():
    class Batch:
        def __init__(self, model, input_ids, warm_cache=None, draft_model=None,
                     draft_kind=None, kv_bits=None, right_pad_per_row=None):
            self.values = (model, input_ids, warm_cache, right_pad_per_row)

    original, normalized, model = object(), object(), object()
    calls = []

    def normalize(cache):
        calls.append(cache)
        return normalized

    patch_prompt_batch(Batch, normalize=normalize, accepts_model=lambda m: m is model)
    batch = Batch(model, [[7]], original, right_pad_per_row=[0])
    assert batch.values == (model, [[7]], normalized, [0])
    assert calls == [original]
    patch_prompt_batch(Batch, normalize=normalize, accepts_model=lambda m: True)
    Batch(model, [[8]], original)
    assert calls == [original, original]  # installation is idempotent


def test_single_warm_hook_rejects_ineligible_requests():
    class Batch:
        def __init__(self, model, input_ids, warm_cache=None, draft_model=None,
                     draft_kind=None, kv_bits=None, right_pad_per_row=None):
            self.cache = warm_cache

    model, cache = object(), object()

    def unexpected(_):
        raise AssertionError('ineligible request reached normalization')

    patch_prompt_batch(Batch, normalize=unexpected, accepts_model=lambda m: m is model)
    for changes in ({'warm_cache': None}, {'model': object()}, {'input_ids': [[1], [2]]},
                    {'draft_model': object()}, {'draft_kind': 'mtp'}, {'kv_bits': 4},
                    {'right_pad_per_row': [1]}):
        args = dict(model=model, input_ids=[[1]], warm_cache=cache)
        args.update(changes)
        assert Batch(**args).cache is args['warm_cache']
