import weakref

from qwen_booster.mtp_lifecycle import patch_batch_class, release_finished_batch


class State:
    pass


class Batch:
    def __init__(self):
        self.live = 1
        self.prompt_cache = [State()]
        self.hidden = self.shared_kv_states = self.prompt_tokens = self.first_tokens = State()
        self._rounds_iter = None

    def __len__(self):
        return self.live

    def next(self):
        self.live = 0
        return ['final-token']

    def filter(self, keep):
        self.live = len(keep)


def test_live_state_preserved_and_finished_generator_closed():
    batch = Batch()
    state = weakref.ref(batch.prompt_cache[0])
    finalized = []

    def rounds(owner):
        try:
            yield owner.prompt_cache
        finally:
            finalized.append(True)

    batch._rounds_iter = rounds(batch)
    next(batch._rounds_iter)
    assert not release_finished_batch(batch)
    assert state() is not None
    batch.live = 0
    assert release_finished_batch(batch)
    assert finalized == [True]
    assert state() is None
    assert not release_finished_batch(batch)


def test_completion_and_cancellation_hooks_preserve_responses_and_apc_owner():
    class Patched(Batch):
        pass

    released = []
    patch_batch_class(Patched, lambda: released.append(True))
    batch = Patched()
    snapshot = list(batch.prompt_cache)
    state = weakref.ref(snapshot[0])
    assert batch.next() == ['final-token']
    assert batch.prompt_cache == []
    assert state() is snapshot[0]
    other = Patched()
    other.filter([0])
    assert other.prompt_cache
    other.filter([])
    assert other.prompt_cache == []
    assert released == [True, True]
