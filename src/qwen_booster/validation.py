"""Real-model cache validation, run separately from performance experiments."""

import gc
import json
import time
from pathlib import Path


def validate(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from .cli import environment, local_model
    from .runtime import Engine

    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    path, _ = local_model(args.model)
    model, tokenizer = load(path)
    engine = Engine(model, prefill_step_size=2048)
    prompts = [
        "请用中文解释为什么合并两个有序数组的时间复杂度是 O(n+m)，并分析空数组和重复元素。",
        "Write a Python binary_search function returning the first matching index. Explain edge cases.",
    ]

    def run(iterator):
        mx.synchronize()
        start = time.perf_counter()
        first = None
        tokens = []
        try:
            for token, _ in iterator:
                if first is None:
                    first = time.perf_counter()
                tokens.append(int(token))
        finally:
            iterator.close()
            mx.synchronize()
        return tokens, first - start

    cases = []
    for prompt in prompts:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        native, _ = run(generate_step(mx.array(tokens), model, max_tokens=32))
        cold, cold_ttft = run(engine.generate(tokens, max_tokens=32))
        warm, warm_ttft = run(engine.generate(tokens, max_tokens=32))
        cached = engine.last_cached_tokens
        # A second restore catches aliasing/mutation of the stored recurrent state.
        restored, _ = run(engine.generate(tokens, max_tokens=32))
        cases.append({
            "prompt": prompt, "prompt_tokens": len(tokens), "cached_tokens": cached,
            "native_cold_equal": native == cold,
            "native_warm_equal": native == warm,
            "repeat_restore_equal": native == restored,
            "cold_ttft_seconds": cold_ttft, "warm_ttft_seconds": warm_ttft,
            "output_token_ids": native, "output_text": tokenizer.decode(native),
        })
        print(json.dumps(cases[-1], ensure_ascii=False), flush=True)
        # Test an extension that shares the snapshot, and a divergent prompt.
        for variant in (tokens + native + [tokens[-1]], tokens + native[:8], tokens[:len(tokens)//2] + native[:8]):
            reference, _ = run(generate_step(mx.array(variant), model, max_tokens=16))
            actual, _ = run(engine.generate(variant, max_tokens=16))
            cases.append({"prompt_tokens": len(variant), "cached_tokens": engine.last_cached_tokens,
                          "native_cold_equal": reference == actual,
                          "native_warm_equal": True, "repeat_restore_equal": True,
                          "case": "extension_or_divergence"})
        gc.collect()
    passed = all(c[flag] for c in cases for flag in (
        "native_cold_equal", "native_warm_equal", "repeat_restore_equal"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": passed, "environment": environment(), "cases": cases},
                                 ensure_ascii=False, indent=2) + "\n")
    if not passed:
        raise ValueError(f"Cache parity failed; see {output}")
    print(f"Cache parity passed: {len(cases)} cases", flush=True)


def validate_lookup(args):
    import statistics
    import mlx.core as mx
    from mlx_lm import load
    from .cli import environment, local_model
    from .lookup import lookup_generate
    from .runtime import Engine

    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    path, _ = local_model(args.model)
    model, tokenizer = load(path)
    engine = Engine(model)
    prompts = [
        "请解释二分查找的原理，并用 Python 实现。",
        "Copy the following Python code exactly, without explanations:\n"
        "def merge_sorted(left, right):\n    output = []\n    i = j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n            output.append(left[i])\n            i += 1\n"
        "        else:\n            output.append(right[j])\n            j += 1\n"
        "    return output + left[i:] + right[j:]\n",
    ]
    cases = []
    for prompt in prompts:
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
        engine.prepare(tokens)
        rows = []
        # First pair warms both execution paths, then alternate order.
        for repeat in range(4):
            outputs = {}
            for mode in (["native", "lookup"] if repeat % 2 == 0 else ["lookup", "native"]):
                stats = {}
                cache = engine.prepare(tokens)
                mx.synchronize()
                started = time.perf_counter()
                if mode == "native":
                    from mlx_lm.generate import generate_step
                    result = [int(t) for t, _ in generate_step(mx.array(tokens[-1:]), model,
                              max_tokens=128, prompt_cache=cache)]
                else:
                    result = list(lookup_generate(model, tokens, cache, max_tokens=128, stats=stats))
                mx.synchronize()
                elapsed = time.perf_counter() - started
                outputs[mode] = result
                if repeat:
                    rows.append({"mode": mode, "repeat": repeat, "seconds": elapsed, "stats": stats})
            if outputs["native"] != outputs["lookup"]:
                cases.append({"prompt": prompt, "passed": False, "rows": rows,
                              "native_ids": outputs["native"], "lookup_ids": outputs["lookup"]})
                break
        else:
            native_time = statistics.median(r["seconds"] for r in rows if r["mode"] == "native")
            lookup_time = statistics.median(r["seconds"] for r in rows if r["mode"] == "lookup")
            cases.append({"prompt": prompt, "passed": True, "rows": rows,
                          "speedup": native_time / lookup_time})
        print(json.dumps({k: v for k, v in cases[-1].items() if k != "rows"}, ensure_ascii=False), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    passed = all(c["passed"] for c in cases)
    output.write_text(json.dumps({"passed": passed, "environment": environment(), "cases": cases,
        "scope": "greedy, warm exact-prefix cache, 128 fixed tokens, EOS ignored; experimental only"},
        ensure_ascii=False, indent=2) + "\n")
    if not passed:
        raise ValueError(f"Prompt lookup parity failed; keep disabled. See {output}")
