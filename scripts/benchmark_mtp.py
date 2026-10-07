"""Paired native-MTP screen using the SAME target weights inside MLX-VLM.

Run in the isolated MLX-VLM environment after other GPU workloads finish.
Fresh request caches; model and draft resident; greedy, real EOS stopping.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


PROMPTS = [
    "请解释二分查找为什么能达到对数复杂度，分析重复元素和空数组，并给出 Python 实现。",
    "Implement a Python LRU cache using only the standard library. Include edge cases and tests.",
    "Design an incremental CSV parser handling quoted commas and escaped quotes. Explain the state transitions.",
]


def summarize_pairs(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["case"], pair["block"])].append(pair)
    summary = []
    for (case, block), rows in sorted(groups.items()):
        passed = all(r["token_parity"] for r in rows)
        summary.append({"case": case, "block": block, "pairs": len(rows),
            "token_parity": passed,
            "median_paired_speedup": statistics.median(r["end_to_end_speedup"] for r in rows)
                if passed else None})
    return summary


def main():
    import mlx.core as mx
    from mlx_vlm import load, stream_generate
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-mb", type=int, default=64000,
                        help="Explicit MLX memory budget; increase for the larger Flash-Next target")
    parser.add_argument("--allocator-cache-mb", type=int, default=2000)
    parser.add_argument("--thinking", action="store_true", help="Compare the native thinking template")
    args = parser.parse_args()
    if (args.tokens < 2 or args.repeats < 1 or any(b < 2 for b in args.blocks)
            or min(args.memory_mb, args.allocator_cache_mb) <= 0):
        parser.error("tokens/blocks must be >=2, repeats >=1 and memory budgets positive")
    for path in (args.model, args.draft):
        if not Path(path).is_dir():
            parser.error("model and draft must be local directories")
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite results")
    from qwen_booster.models import resolve_local_model
    target_path, target_config, profile = resolve_local_model(args.model)
    draft_config = json.loads((Path(args.draft) / 'config.json').read_text())
    if draft_config.get('model_type') != profile.draft_type:
        parser.error(f"target requires draft model_type={profile.draft_type}")
    mx.set_memory_limit(args.memory_mb * 1_000_000)
    mx.set_cache_limit(args.allocator_cache_mb * 1_000_000)
    model, processor = load(target_path)
    draft, kind = load_drafter(args.draft)
    if kind != "mtp":
        raise ValueError(f"expected native MTP, got {kind}")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    mx.eval(model.parameters(), draft.parameters())
    mx.synchronize()

    def run(prompt, block):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = speculative_stats_snapshot(draft)
        extra = {"draft_model": draft, "draft_kind": kind, "draft_block_size": block} if block else {}
        iterator = stream_generate(model, processor, prompt, max_tokens=args.tokens,
            temperature=0, prefill_step_size=2048, **extra)
        started = time.perf_counter()
        arrivals, text, last = [], "", None
        try:
            for response in iterator:
                if response.token is not None and response.generation_tokens > len(arrivals):
                    arrivals.append(time.perf_counter() - started)
                text += response.text
                last = response
        finally:
            iterator.close()
            mx.synchronize()
        elapsed = time.perf_counter() - started
        rounds, accepted, drafted = speculative_stats_since(draft, before)
        decode = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0
        return {"block_size": block, "seconds": elapsed,
            "ttft_seconds": arrivals[0] if arrivals else None,
            "decode_tps": (len(arrivals)-1)/decode if decode else None,
            "token_ids": last.token_ids, "text": text, "finish_reason": last.finish_reason,
            "prompt_tokens": last.prompt_tokens, "generation_tokens": last.generation_tokens,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "draft_rounds": rounds, "draft_accepted": accepted, "draft_proposed": drafted}

    pairs = []
    with output.open("x") as handle:
        metadata = {"type": "metadata", "scope": __doc__,
            "enable_thinking": args.thinking,
            "model_profile": profile.id, "model_path": str(target_path),
            "draft_path": str(Path(args.draft).resolve()),
            "memory_mb": args.memory_mb, "allocator_cache_mb": args.allocator_cache_mb,
            "ple_storage": target_config.get('text_config', target_config).get('ple_storage'),
            "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm", "transformers")},
            "target_config_sha256": hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest(),
            "draft_config_sha256": hashlib.sha256((Path(args.draft)/"config.json").read_bytes()).hexdigest()}
        handle.write(json.dumps(metadata) + "\n")
        handle.flush()
        for index, text in enumerate(PROMPTS):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_hash = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
            for block in args.blocks:
                for repeat in range(args.repeats + 1):
                    outputs = {}
                    for mode in ([0, block] if repeat % 2 == 0 else [block, 0]):
                        row = run(prompt, mode)
                        row["prompt_token_sha256"] = prompt_hash
                        if row["prompt_tokens"] != len(prompt_ids):
                            raise RuntimeError("backend prompt token count differs from the comparison fixture")
                        outputs[mode] = row
                        row.update(type="measurement", case=index, repeat=repeat, warmup=repeat == 0)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                    equal = outputs[0]["token_ids"] == outputs[block]["token_ids"]
                    pair = {"case": index, "block": block, "repeat": repeat,
                            "token_parity": equal,
                            "end_to_end_speedup": outputs[0]["seconds"] / outputs[block]["seconds"]}
                    print(json.dumps(pair), flush=True)
                    if repeat:
                        pairs.append(pair)
        report = {"pairs": pairs, "all_greedy_equal": all(p["token_parity"] for p in pairs),
                  "per_case_block": summarize_pairs(pairs),
                  "scope": "short prompts only; per-case/block results required before enabling a default"}
        output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["all_greedy_equal"]:
            raise RuntimeError("MTP greedy output differs; inspect pairs before enabling")


if __name__ == "__main__":
    main()
