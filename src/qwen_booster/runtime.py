"""Exact-prefix snapshots for hybrid attention models.

Snapshots contain *all* layer state, including recurrent and convolution state.
An unmatched prefix is recomputed; recurrent caches are never truncated as KV.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass

from .hybrid_cache import HybridPrefixCache, final_block_hash


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

    def _fresh_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def prepare(self, tokens, reuse=True):
        import mlx.core as mx

        if not tokens:
            raise ValueError("prompt must be nonempty")
        count, cache = self.prefixes.get(tokens) if reuse else (0, None)
        self.last_cached_tokens = count
        if cache is None:
            cache = self._fresh_cache()
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


class HybridEngine(Engine):
    """Block-aligned checkpoint engine following vLLM hybrid-cache semantics.

    Recurrent state is never trimmed: restores come from block-aligned
    checkpoints and the engine replays at most one block plus the kept final
    token. ``commit`` publishes a post-generation checkpoint only when the
    finished length is itself block aligned; otherwise the next request
    replays the unaligned tail from the newest boundary.
    """

    def __init__(self, model, prefill_step_size=2048, cache_bytes=2_000_000_000,
                 active_memory_budget=None, block_size=256, disk_root=None,
                 disk_bytes=0, disk_entries=64, disk_save=None, disk_load=None):
        super().__init__(model, prefill_step_size=prefill_step_size,
                         cache_bytes=cache_bytes,
                         active_memory_budget=active_memory_budget)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if bool(disk_root) != (disk_bytes > 0):
            raise ValueError("disk checkpointing requires disk_root and positive disk_bytes")
        self.block_size = block_size
        self.prefixes = HybridPrefixCache(max_bytes=cache_bytes, block_size=block_size)
        self.disk = None
        if disk_root:
            from .apfs import CheckpointDiskStore
            self.disk = CheckpointDiskStore(disk_root, disk_bytes,
                                            max_entries=disk_entries,
                                            save=disk_save, load=disk_load)

    def _offsets_match(self, cache, count):
        from mlx_lm.models.cache import KVCache
        offsets = [c.offset for c in cache if isinstance(c, KVCache)]
        return bool(offsets) and all(offset == count for offset in offsets)

    def _lookup(self, tokens):
        count, cache = self.prefixes.find(tokens)
        if self.disk is not None:
            disk_count, digest = self.disk.match(tokens, self.block_size)
            if disk_count > count:
                disk_cache = self.disk.get(digest)
                if disk_cache is not None:
                    count, cache = disk_count, disk_cache
        if cache is not None and not self._offsets_match(cache, count):
            return 0, None
        return count, cache

    def _store(self, tokens, cache):
        size = sum(c.nbytes for c in cache)
        stored = self.prefixes.store(tokens, cache)
        if self.disk is not None:
            self.disk.put(final_block_hash(tokens, self.block_size), len(tokens),
                          cache, nbytes=size)
        return stored

    def prepare(self, tokens, reuse=True):
        import mlx.core as mx

        if not tokens:
            raise ValueError("prompt must be nonempty")
        count, cache = self._lookup(tokens) if reuse else (0, None)
        self.last_cached_tokens = count
        if cache is None:
            cache = self._fresh_cache()
        end = len(tokens) - 1
        boundary = end - end % self.block_size
        processed, stored = count, False
        while processed < end:
            stop = min(processed + self.prefill_step_size, end)
            if reuse and not stored and count < boundary < stop:
                stop = boundary
            chunk = mx.array(tokens[processed:stop], dtype=mx.uint32)
            self.model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            self.check_memory()
            processed = stop
            if reuse and not stored and boundary > count and processed == boundary:
                self._store(tokens[:boundary], cache)
                stored = True
        return cache

    def commit(self, tokens, cache):
        """Publish only block-aligned finished prefixes; unaligned tails replay."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, ArraysCache

        if not cache or len(tokens) % self.block_size:
            return False
        if any(not isinstance(c, (KVCache, ArraysCache)) for c in cache):
            return False
        offsets = [c.offset for c in cache if isinstance(c, KVCache)]
        if not offsets or any(offset != len(tokens) for offset in offsets):
            return False
        mx.eval([c.state for c in cache])
        return self._store(tokens, cache)


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
