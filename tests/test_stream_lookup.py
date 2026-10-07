from qwen_booster import lookup
from qwen_booster.runtime import stream_lookup


class Detokenizer:
    def __init__(self):
        self.pending = ""

    def add_token(self, token):
        self.pending += str(token)

    @property
    def last_segment(self):
        result, self.pending = self.pending, ""
        return result

    def finalize(self):
        pass


class Tokenizer:
    eos_token_ids = {9}

    @property
    def detokenizer(self):
        return Detokenizer()


def test_stream_stops_at_eos_and_closes_underlying_generator(monkeypatch):
    closed = []
    def tokens(*args, **kwargs):
        try:
            yield from [1, 2, 9, 3]
        finally:
            closed.append(True)
    monkeypatch.setattr(lookup, "lookup_generate", tokens)
    chunks = list(stream_lookup(None, Tokenizer(), [0], [], 4))
    assert "".join(c.text for c in chunks) == "12"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].generation_tokens == 3
    assert closed == [True]


def test_stream_flushes_final_token_and_handles_cancellation(monkeypatch):
    closed = []
    def tokens(*args, **kwargs):
        try:
            yield from [1, 2]
        finally:
            closed.append(True)
    monkeypatch.setattr(lookup, "lookup_generate", tokens)
    chunks = list(stream_lookup(None, Tokenizer(), [0], [], 2))
    assert "".join(c.text for c in chunks) == "12"
    assert chunks[-1].finish_reason == "length"
    iterator = stream_lookup(None, Tokenizer(), [0], [], 2)
    next(iterator)
    iterator.close()
    assert closed == [True, True]
