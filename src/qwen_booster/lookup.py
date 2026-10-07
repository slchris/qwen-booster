"""Greedy prompt lookup with full-state rollback for hybrid Qwen models.

Experimental: rejection restores every layer, then replays only accepted inputs.
This trades extra work on misses for correctness of recurrent state. Off by default.
"""

import copy
import time


def draft_from_history(history, max_draft=3, min_match=3, max_match=16):
    """Find a past continuation of the current suffix, longest match first."""
    for size in range(min(max_match, len(history) - 1), min_match - 1, -1):
        suffix = history[-size:]
        for start in range(len(history) - size - 1, -1, -1):
            if history[start:start + size] == suffix:
                # Only propose already observed tokens; never read beyond history.
                return history[start + size:min(start + size + max_draft, len(history))]
    return []


def accepted_prefix(draft, target):
    accepted = 0
    for candidate, verified in zip(draft, target):
        if candidate != verified:
            break
        accepted += 1
    return accepted


def lookup_generate(model, prompt, cache, max_tokens=128, max_draft=3, stats=None):
    """Cache must represent prompt[:-1]. Produces exactly max_tokens (EOS ignored).

    This experimental path supports greedy generation only. The cache is private
    to this iterator, so a rejected speculative step cannot corrupt PrefixStore.
    """
    import mlx.core as mx
    if not prompt or max_tokens < 1 or max_draft < 1:
        raise ValueError("nonempty prompt and positive token limits required")
    stats = stats if stats is not None else {}
    stats.update(rounds=0, offered=0, accepted=0, replay_tokens=0, verify_seconds=0.0,
                 replay_seconds=0.0, snapshot_seconds=0.0, disabled=False)
    history = list(prompt)
    produced = 0
    failed_rounds = 0
    while produced < max_tokens:
        draft = [] if stats["disabled"] else draft_from_history(
            history, max_draft=min(max_draft, max_tokens - produced - 1))
        started = time.perf_counter()
        checkpoint = copy.deepcopy(cache) if draft else None
        if checkpoint is not None:
            mx.eval([c.state for c in checkpoint])
        stats["snapshot_seconds"] += time.perf_counter() - started
        inputs = [history[-1]] + draft
        started = time.perf_counter()
        logits = model(mx.array([inputs], dtype=mx.uint32), cache=cache)
        choices = mx.argmax(logits[0], axis=-1)
        mx.eval(choices, [c.state for c in cache])
        target = choices.tolist()
        stats["verify_seconds"] += time.perf_counter() - started
        count = accepted_prefix(draft, target)
        emitted = target[:count + 1]
        stats["rounds"] += 1
        stats["offered"] += len(draft)
        stats["accepted"] += count
        if count < len(draft):
            cache = checkpoint
            started = time.perf_counter()
            model(mx.array([inputs[:count + 1]], dtype=mx.uint32), cache=cache)
            mx.eval([c.state for c in cache])
            stats["replay_seconds"] += time.perf_counter() - started
            stats["replay_tokens"] += count + 1
        if draft:
            failed_rounds = failed_rounds + 1 if count == 0 else 0
            if failed_rounds >= 3:
                stats["disabled"] = True
        history.extend(emitted)
        produced += len(emitted)
        yield from emitted
