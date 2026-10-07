"""Build deterministic long-context retrieval and cross-record reasoning cases.

Uses the deployed tokenizer/template, reserves output within each context tier,
and records actual token counts. Synthetic diagnostics must be supplemented by
real code/document task evaluation before choosing a daily-use context tier.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    from qwen_booster.cli import local_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[8192, 16384, 32768, 65536, 131072, 262144])
    parser.add_argument("--output", required=True)
    parser.add_argument("--headroom-tokens", type=int, default=128,
                        help="Leave extra space for follow-up turns in session fixtures (at least 128)")
    args = parser.parse_args()
    if args.model:
        from qwen_booster.models import resolve_local_model
        path, config, _ = resolve_local_model(args.model, require_weights=False)
    else:
        path, config = local_model(None)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    native = config.get("text_config", config)["max_position_embeddings"]
    if args.headroom_tokens < 128 or any(n <= args.headroom_tokens + 512 for n in args.lengths):
        parser.error("headroom must be >=128 and leave at least 512 input tokens in each tier")
    rng = random.Random(1729)
    # Varied records avoid measuring a single repeating-text copy workload.
    filler = "\n".join(f"记录 {i:06d}: 项目编号 {rng.randrange(10000000, 99999999)}; "
                       f"状态 {rng.choice(['待复核', '已归档', '处理中'])}; 校验码 {rng.getrandbits(48):012x}."
                       for i in range(18000))
    unit = tokenizer.encode(filler, add_special_tokens=False, verbose=False)
    with Path(args.output).open("x") as out:
        for length in args.lengths:
            if not 1024 <= length <= native:
                raise ValueError(f"invalid native tier {length}")
            for position in (0.05, 0.5, 0.95):
                secret = f"K-{rng.getrandbits(48):012x}"
                fact = f"\n唯一有效记录：项目青鹭的负责人是林舟。林舟的访问口令是 {secret}。\n"
                # Put linked facts apart to require cross-record association.
                if position == 0.5:
                    fact = "\n唯一有效记录：项目青鹭的负责人是林舟。\n"
                    tail = f"\n人员名册：林舟的访问口令是 {secret}。\n"
                else:
                    tail = ""
                question = "\n根据有效记录与人员名册，项目青鹭负责人的访问口令是什么？只输出口令。"

                def make(n):
                    split = int(n * position)
                    content = (tokenizer.decode(unit[:split]) + fact
                               + tokenizer.decode(unit[split:n]) + tail + question)
                    messages = [{"role": "user", "content": content}]
                    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, enable_thinking=False, return_dict=False)
                    return messages, tokens

                low, high = 0, min(len(unit), length)
                while low < high:
                    mid = (low + high + 1) // 2
                    if len(make(mid)[1]) + args.headroom_tokens <= length:
                        low = mid
                    else:
                        high = mid - 1
                messages, tokens = make(low)
                assert len(tokens) + args.headroom_tokens <= length
                case = {"id": f"context-{length}-position-{position}", "messages": messages,
                        "max_tokens": 128, "enable_thinking": False,
                        "expected_substrings": [secret], "context_tier": length,
                        "prompt_tokens": len(tokens), "fact_position_fraction": position,
                        "headroom_tokens": args.headroom_tokens,
                        "prompt_token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        "scope": "synthetic retrieval/cross-record diagnostic; not a general quality benchmark"}
                out.write(json.dumps(case, ensure_ascii=False) + "\n")
                print(case["id"], len(tokens), flush=True)


if __name__ == "__main__":
    main()
