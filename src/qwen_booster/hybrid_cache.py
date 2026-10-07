"""vLLM-style block-aligned prefix caching for hybrid attention models.

The legacy PrefixStore keeps one exact snapshot per request and matches by
scanning every stored token prefix. This module follows vLLM's automatic
prefix caching instead:

- a prefix is identified by a chain of fixed-size token blocks where each
  block hash is hash(parent_hash, block_token_ids);
- only block-aligned prefixes are stored, so lookups never need to truncate
  attention KV or recurrent/convolution state to a partial position;
- hybrid (GDN/convolution) state cannot be rolled back, so a hit restores the
  newest checkpoint at or before the matched prefix and the caller recomputes
  the remaining suffix. Recompute is bounded by one block per request.

The index is model-independent and tested without a GPU. Snapshot isolation
relies on the caller cloning state; production clones come from the engine.
"""

import copy
import hashlib
import struct
from collections import OrderedDict


def token_block_bytes(block):
    return struct.pack(f"<{len(block)}I", *block)


def block_hash(parent, block):
    """Chained block identity, the same shape vLLM uses for prefix blocks."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(parent.to_bytes(8, "little"))
    digest.update(token_block_bytes(block))
    return int.from_bytes(digest.digest(), "little")


def block_chain(tokens, block_size):
    """Yield (block_end, hash) for every complete block, in order."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    parent = 0
    for end in range(block_size, len(tokens) + 1, block_size):
        parent = block_hash(parent, tokens[end - block_size:end])
        yield end, parent


def final_block_hash(tokens, block_size):
    """Chained hash of the last complete block; the only legal store key."""
    if not tokens or len(tokens) % block_size:
        raise ValueError("checkpoints must be nonempty and block aligned")
    for end, digest in block_chain(tokens, block_size):
        pass
    return digest


class HybridPrefixCache:
    """Byte-budgeted LRU over block-aligned hybrid-state checkpoints.

    Standard stats keys match PrefixStore so existing telemetry keeps working.
    """

    class Entry:
        __slots__ = ("tokens_len", "nbytes", "cache")

        def __init__(self, tokens_len, nbytes, cache):
            self.tokens_len = tokens_len
            self.nbytes = nbytes
            self.cache = cache

    def __init__(self, max_bytes=2_000_000_000, block_size=256, max_entries=16):
        if max_bytes < 0 or block_size <= 0 or max_entries < 0:
            raise ValueError("cache budgets and block size must be valid")
        self.max_bytes = max_bytes
        self.block_size = block_size
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.skipped = 0

    @property
    def nbytes(self):
        return sum(entry.nbytes for entry in self.entries.values())

    def stats(self):
        return {"entries": len(self.entries), "bytes": self.nbytes,
                "budget_bytes": self.max_bytes, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions,
                "skipped_oversize_or_disabled": self.skipped,
                "block_size": self.block_size}

    def clear(self):
        self.entries.clear()

    def find(self, tokens):
        """Longest stored block-aligned prefix strictly shorter than tokens.

        Chained hashes make every block entry self-validating, so gaps between
        stored boundaries are legal and the walk continues past them. The
        final token is always left for the caller so generation logits are
        computed fresh.
        """
        best = None
        for end, digest in block_chain(tokens, self.block_size):
            if end >= len(tokens):
                break
            entry = self.entries.get(digest)
            if entry is not None and entry.tokens_len == end:
                best = (end, digest)
        if best is None:
            self.misses += 1
            return 0, None
        end, digest = best
        self.hits += 1
        self.entries.move_to_end(digest)
        return end, copy.deepcopy(self.entries[digest].cache)

    def store(self, tokens, cache):
        """Insert a checkpoint; tokens must be nonempty and block aligned."""
        if not tokens or len(tokens) % self.block_size:
            raise ValueError("store requires a nonempty block-aligned prefix")
        size = sum(c.nbytes for c in cache)
        if size > self.max_bytes or self.max_entries == 0:
            self.skipped += 1
            return False
        digest = final_block_hash(tokens, self.block_size)
        self.entries.pop(digest, None)
        while self.entries and (self.nbytes + size > self.max_bytes
                                or len(self.entries) >= self.max_entries):
            self.entries.popitem(last=False)
            self.evictions += 1
        self.entries[digest] = self.Entry(len(tokens), size, copy.deepcopy(cache))
        return True
