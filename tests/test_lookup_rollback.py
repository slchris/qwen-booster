"""Exercise rejection/replay against an independent serial recurrent oracle."""

import sys
import types

import numpy as np
import pytest

from qwen_booster import lookup


class RecurrentCache:
    def __init__(self, state=0):
        self.state = state


def next_state(state, token):
    return (state * 3 + token + 1) % 11


class RecurrentModel:
    def __call__(self, inputs, cache):
        logits = []
        for token in inputs[0]:
            cache[0].state = next_state(cache[0].state, int(token))
            row = np.zeros(11)
            row[(cache[0].state + 2) % 11] = 1
            logits.append(row)
        return np.asarray([logits])


def serial(prompt, count):
    state = 0
    for token in prompt[:-1]:
        state = next_state(state, token)
    result = []
    pending = prompt[-1]
    for _ in range(count):
        state = next_state(state, pending)
        pending = (state + 2) % 11
        result.append(pending)
    return result


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_speculative_rejection_restores_recurrent_state(monkeypatch, accepted):
    mx = types.ModuleType("mlx.core")
    mx.array, mx.argmax, mx.uint32 = np.asarray, np.argmax, np.uint32
    mx.eval = lambda *args: None
    parent = types.ModuleType("mlx")
    parent.core = mx
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    def proposer(history, max_draft):
        proposed = serial(history, max_draft)
        if accepted < len(proposed):
            proposed[accepted] = (proposed[accepted] + 1) % 11
        return proposed

    monkeypatch.setattr(lookup, "draft_from_history", proposer)
    prompt = [1, 2, 3]
    state = 0
    for token in prompt[:-1]:
        state = next_state(state, token)
    stats = {}
    actual = list(lookup.lookup_generate(RecurrentModel(), prompt, [RecurrentCache(state)],
                                        max_tokens=19, max_draft=3, stats=stats))
    assert actual == serial(prompt, 19)
    if accepted < 3:
        assert stats["replay_tokens"] > 0
    else:
        assert stats["replay_tokens"] == 0
    if accepted == 0:
        assert stats["disabled"] is True
