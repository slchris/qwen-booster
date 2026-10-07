"""Exact-prefix snapshots for hybrid attention models.

Snapshots contain *all* layer state, including recurrent and convolution state.
An unmatched prefix is recomputed; recurrent caches are never truncated as KV.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass


class PrefixStore:
    def __init__(self, max_bytes=2_000_000_000, max_entries=4):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("cache budgets must be nonnegative")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.skipped = 0

    def stats(self):
        return {"entries": len(self.entries), "bytes": self.nbytes,
                "budget_bytes": self.max_bytes, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions,
                "skipped_oversize_or_disabled": self.skipped}

    def clear(self):
        self.entries.clear()

    @property
    def nbytes(self):
        return sum(size for _, size in self.entries.values())

    def put(self, tokens, cache):
        key = tuple(tokens)
        size = sum(c.nbytes for c in cache)
        if not key or size > self.max_bytes or self.max_entries == 0:
            self.skipped += 1
            return False
        self.entries.pop(key, None)
        while self.entries and (self.nbytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            self.entries.popitem(last=False)
            self.evictions += 1
        self.entries[key] = (copy.deepcopy(cache), size)
        return True

    def get(self, tokens):
        tokens = tuple(tokens)
        matches = [key for key in self.entries if len(key) < len(tokens) and tokens[:len(key)] == key]
        if not matches:
            self.misses += 1
            return 0, None
        key = max(matches, key=len)
        self.hits += 1
        cache, _ = self.entries[key]
        self.entries.move_to_end(key)
        return len(key), copy.deepcopy(cache)


class Engine:
    """Single-model, single-request executor. Create a separate engine per model."""

    def __init__(self, model, prefill_step_size=2048, cache_bytes=2_000_000_000,
                 active_memory_budget=None):
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be positive")
        self.model = model
        self.prefill_step_size = prefill_step_size
        self.prefixes = PrefixStore(max_bytes=cache_bytes)
        self.last_cached_tokens = 0
        self.active_memory_budget = active_memory_budget

    def check_memory(self):
        if self.active_memory_budget is not None:
            import mlx.core as mx
            if mx.get_active_memory() > self.active_memory_budget:
                raise MemoryError("MLX active memory exceeds configured budget")

    def prepare(self, tokens, reuse=True):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if not tokens:
            raise ValueError("prompt must be nonempty")
        count, cache = self.prefixes.get(tokens) if reuse else (0, None)
        self.last_cached_tokens = count
        if cache is None:
            cache = make_prompt_cache(self.model)
        # Leave the last token for generation, so no logits need to be cached.
        end = len(tokens) - 1
        for start in range(count, end, self.prefill_step_size):
            chunk = mx.array(tokens[start:min(start + self.prefill_step_size, end)], dtype=mx.uint32)
            self.model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache])
            # Match native MLX-LM: varying attention shapes must not accumulate
            # unused allocator buffers across a long prefill.
            mx.clear_cache()
            self.check_memory()
        if reuse and count != end:
            self.prefixes.put(tokens[:end], cache)
        return cache

    def generate(self, tokens, max_tokens=128, reuse=True):
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        cache = self.prepare(tokens, reuse=reuse)
        generated = []
        iterator = generate_step(mx.array(tokens[-1:], dtype=mx.uint32), self.model,
                                 max_tokens=max_tokens, prompt_cache=cache,
                                 prefill_step_size=self.prefill_step_size)
        try:
            for token, logprobs in iterator:
                generated.append(int(token))
                yield token, logprobs
        finally:
            iterator.close()
        if reuse:
            self.commit(tokens + generated, cache)

    def commit(self, tokens, cache):
        """Publish a completed native generation only at its exact consumed boundary.

        MLX-LM 0.32 pipelines one forward ahead of the yielded token. Check all
        attention offsets before associating recurrent state with a token prefix.
        A different generator contract fails closed, without slicing hybrid state.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, ArraysCache

        if not cache or any(not isinstance(c, (KVCache, ArraysCache)) for c in cache):
            return False
        offsets = [c.offset for c in cache if isinstance(c, KVCache)]
        if not offsets or any(offset != len(tokens) for offset in offsets):
            return False
        mx.eval([c.state for c in cache])
        return self.prefixes.put(tokens, cache)


@dataclass
class TextChunk:
    text: str
    generation_tokens: int
    finish_reason: str | None


def stream_lookup(model, tokenizer, tokens, cache, max_tokens):
    """Streaming adapter for the explicitly enabled greedy lookup experiment."""
    from .lookup import lookup_generate
    detokenizer = tokenizer.detokenizer
    iterator = lookup_generate(model, tokens, cache, max_tokens=max_tokens)
    count, reason = 0, "length"
    try:
        for token in iterator:
            count += 1
            if token in tokenizer.eos_token_ids:
                reason = "stop"
                break
            detokenizer.add_token(token)
            if count < max_tokens:
                yield TextChunk(detokenizer.last_segment, count, None)
        detokenizer.finalize()
        yield TextChunk(detokenizer.last_segment, count, reason)
    finally:
        iterator.close()
