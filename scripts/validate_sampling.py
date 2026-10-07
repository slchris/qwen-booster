"""Real-model seeded sampling parity across native, cold-engine and warm-engine paths."""

import argparse
import json
from pathlib import Path


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from qwen_booster.cli import local_model, environment
    from qwen_booster.runtime import Engine
    from qwen_booster.sampling import settings, native_options

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("refusing to overwrite sampling results")
    path, _ = local_model(args.model)
    model, tokenizer = load(path)
    engine = Engine(model)
    rows = []
    for thinking in (False, True):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content":
            "请解释二分查找，并分析重复元素、空数组和索引边界。"}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking)
        for config in ({"temperature": 0, "presence_penalty": 1.5, "repetition_penalty": 1.1},
                       {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                        "presence_penalty": 1.5, "seed": 1729}):
            options = settings(config)
            outputs = {}
            for mode in ("native", "cold", "warm"):
                if mode == "native":
                    cache, tokens = None, prompt
                else:
                    if mode == "cold":
                        engine.prefixes.clear()
                    cache, tokens = engine.prepare(prompt), prompt[-1:]
                kwargs = native_options(options, prompt, 48, prefilled=mode != "native")
                iterator = generate_step(mx.array(tokens), model, max_tokens=48,
                                         prompt_cache=cache, **kwargs)
                try:
                    outputs[mode] = [int(t) for t, _ in iterator]
                finally:
                    iterator.close()
                    mx.synchronize()
            row = {"thinking": thinking, "settings": options, "output_ids": outputs,
                   "passed": outputs["native"] == outputs["cold"] == outputs["warm"]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "output_ids"}), flush=True)
    report = {"environment": environment(), "cases": rows,
              "passed": all(r["passed"] for r in rows),
              "scope": "fixed seed implementation parity; not a distributional quality assessment"}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("sampling parity failed; inspect output IDs")


if __name__ == "__main__":
    main()
