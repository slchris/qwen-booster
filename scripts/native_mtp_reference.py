"""Record native MLX-LM outputs for the exact MTP benchmark prompt set."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from benchmark_mtp import PROMPTS


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from qwen_booster.cli import environment, local_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("tokens must be at least 2")
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite output")
    path, config = local_model(args.model)
    mx.set_cache_limit(2_000_000_000)
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "environment": environment(),
            "enable_thinking": args.thinking,
            "config": config, "scope": "cross-backend token parity reference; timings are not a paired performance comparison"}) + "\n")
        for index, prompt in enumerate(PROMPTS):
            tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking)
            iterator = generate_step(mx.array(tokens), model, max_tokens=args.tokens, prefill_step_size=2048)
            started, generated, arrivals = time.perf_counter(), [], []
            try:
                for token, _ in iterator:
                    generated.append(int(token))
                    arrivals.append(time.perf_counter() - started)
                    if token in tokenizer.eos_token_ids:
                        break
            finally:
                iterator.close()
                mx.synchronize()
            row = {"type": "measurement", "case": index, "prompt_tokens": len(tokens),
                "prompt_token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                "token_ids": generated, "text": tokenizer.decode(generated),
                "ttft_seconds": arrivals[0], "seconds": time.perf_counter()-started}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"case": index, "output_tokens": len(generated)}), flush=True)


if __name__ == "__main__":
    main()
