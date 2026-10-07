"""Controlled cross-file semantics embedded in long, varied source context.

This supplements retrieval checks, not a real-repository coding benchmark.
Expected results are computed directly from the fixture's Python semantics.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="Local tokenizer/config for the model under test")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768])
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from qwen_booster.cli import local_model
    from qwen_booster.context import ContextBudget

    if args.model:
        from qwen_booster.models import resolve_local_model
        path, config, _ = resolve_local_model(args.model, require_weights=False)
    else:
        path, config = local_model(None)
    native = config.get("text_config", config)["max_position_embeddings"]
    budget = ContextBudget.from_config(config, native)
    if any(n < 4096 or n > native for n in args.lengths):
        parser.error("lengths must be between 4096 and the native limit")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    rng = random.Random(9821)
    filler = "\n".join(
        f"# file: auxiliary/entry_{i:05d}.py\n"
        f"OFFSET_{i} = {rng.randrange(100, 999)}\n"
        f"def transform_{i}(value):\n    return (value + OFFSET_{i}) % {rng.randrange(11, 97)}\n"
        for i in range(max(args.lengths) // 20 + 1))
    unit = tokenizer.encode(filler, add_special_tokens=False, verbose=False)
    snippets = [
        "\n# file: billing/config.py\nFACTOR = 3\n\n",
        "\n# file: billing/normalize.py\ndef normalize(x):\n    return max(0, x - 2)\n\n",
        "\n# file: billing/service.py\nfrom .config import FACTOR\nfrom .normalize import normalize\n"
        "def bill(xs):\n    return sum(normalize(x) * FACTOR for x in xs if x % 2 == 0)\n\n",
    ]
    question = ("\n任务：只分析 billing 包，auxiliary 包没有被导入。调用方先把输入去重再调用 bill。"
                "输入为 [2,4,4,5,8,-2]。仅返回 JSON：total 是结果，"
                "contributing_inputs 是产生正贡献的去重输入升序列表。")
    values = sorted(set([2, 4, 4, 5, 8, -2]))
    expected = {"total": sum(max(0, x - 2) * 3 for x in values if x % 2 == 0),
                "contributing_inputs": [x for x in values if x % 2 == 0 and max(0, x - 2) > 0]}
    with Path(args.output).open("x") as handle:
        for length in args.lengths:
            for thinking in (False, True):
                reserve = 2048 if thinking else 512

                def make(n):
                    points = [int(n * f) for f in (0.05, 0.5, 0.95)]
                    parts, previous = [], 0
                    for point, snippet in zip(points, snippets):
                        parts.extend([tokenizer.decode(unit[previous:point]), snippet])
                        previous = point
                    parts.extend([tokenizer.decode(unit[previous:n]), question])
                    messages = [{"role": "user", "content": "".join(parts)}]
                    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                        add_generation_prompt=True, enable_thinking=thinking, reasoning_effort="medium")
                    return messages, tokens

                low, high = 0, min(len(unit), length)
                while low < high:
                    mid = (low + high + 1) // 2
                    if len(make(mid)[1]) + reserve <= length:
                        low = mid
                    else:
                        high = mid - 1
                messages, tokens = make(low)
                budget.admit(len(tokens), reserve)
                if len(tokens) + reserve > length:
                    raise ValueError("fixture exceeds its test tier")
                case = {"id": f"long-code-{length}-thinking-{thinking}", "messages": messages,
                    "context_tier": length, "prompt_tokens": len(tokens), "max_tokens": reserve,
                    "prompt_token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    "enable_thinking": thinking, "reasoning_effort": "medium", "seed": 1729,
                    "temperature": 1.0 if thinking else 0.7, "top_p": 0.95 if thinking else 0.8,
                    "top_k": 20, "presence_penalty": 0.0 if thinking else 1.5,
                    "grading": {"kind": "json", "expected": expected}, "scope": __doc__}
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"id": case["id"], "prompt_tokens": len(tokens)}), flush=True)


if __name__ == "__main__":
    main()
