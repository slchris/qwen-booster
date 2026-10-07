"""Synchronized block diagnostics, not an end-to-end speed benchmark."""

import collections
import json
import statistics
import time
from pathlib import Path


def profile(args):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from .cli import environment, local_model, prompt_tokens

    path, _ = local_model(args.model)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    model, tokenizer = load(path)
    mx.set_memory_limit(64_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    records = []
    phase = "warmup"
    chunk_start, chunk_tokens = 0, 0

    class TimedBlock(nn.Module):
        def __init__(self, inner, label, index):
            super().__init__()
            self.inner, self.label, self.index = inner, label, index

        def __call__(self, *inputs, **kwargs):
            mx.eval(inputs[0])
            mx.synchronize()
            start = time.perf_counter()
            result = self.inner(*inputs, **kwargs)
            mx.eval(result)
            cache = kwargs.get("cache")
            if cache is None and len(inputs) >= 3:
                cache = inputs[2]
            if cache is not None:
                mx.eval(cache.state)
            mx.synchronize()
            if phase != "warmup":
                records.append({"phase": phase, "block": self.label, "layer": self.index,
                                "chunk_start": chunk_start, "chunk_tokens": chunk_tokens,
                                "seconds": time.perf_counter() - start})
            return result

    for index, layer in enumerate(model.layers):
        names = ["mlp", "linear_attn" if layer.is_linear else "self_attn"]
        for name in names:
            setattr(layer, name, TimedBlock(getattr(layer, name), name, index))
    tokens = mx.array([prompt_tokens(tokenizer, args.length)])
    for repeat in range(3):
        phase = "warmup" if repeat == 0 else "prefill"
        cache = make_prompt_cache(model)
        for start in range(0, args.length - 1, args.step):
            end = min(start + args.step, args.length - 1)
            chunk_start, chunk_tokens = start, end - start
            model(tokens[:, start:end], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        chunk_start, chunk_tokens = args.length - 1, 1
        logits = model(tokens[:, -1:], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token, [c.state for c in cache])
        phase = "warmup" if repeat == 0 else "decode"
        for _ in range(4):
            chunk_start += 1
            token = mx.argmax(model(token[:, None], cache=cache)[:, -1, :], axis=-1)
            mx.eval(token, [c.state for c in cache])
    groups = collections.defaultdict(list)
    for row in records:
        groups[(row["phase"], row["block"])].append(row["seconds"])
    summary = [{"phase": phase, "block": name, "calls": len(values),
                "sum_seconds": sum(values), "median_call_seconds": statistics.median(values)}
               for (phase, name), values in sorted(groups.items())]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"environment": environment(), "prompt_tokens": args.length,
        "prefill_step_size": args.step,
        "scope": "Synchronized block diagnostics; fences perturb execution. Excludes embedding/head/norm overhead.",
        "summary": summary, "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
