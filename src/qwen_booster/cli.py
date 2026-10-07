"""Local-only benchmark CLI; model files are never downloaded implicitly."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from .metrics import recommend, summarize

MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
CORPUS = """请分析下面的 Python 函数，解释其行为并指出边界条件。
def merge_sorted(left, right):
    output = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            output.append(left[i]); i += 1
        else:
            output.append(right[j]); j += 1
    return output + left[i:] + right[j:]
Explain correctness, duplicates, empty inputs and time complexity.
"""


def local_model(value=None):
    if value:
        path = Path(value).expanduser().resolve()
    else:
        root = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--"))
        ref = root / "refs/main"
        if not ref.exists():
            raise ValueError("No cached target model. Supply --model /path/to/Qwen3.8-27B.")
        path = root / "snapshots" / ref.read_text().strip()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    if (config.get("model_type") != "qwen3_5" or text.get("num_hidden_layers") != 64
            or text.get("hidden_size") != 5120 or text.get("num_experts", 0)):
        raise ValueError("Expected the dense Qwen3.8-27B compatible configuration.")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values()) if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Missing model shards: {missing}")
    return path, config


def environment():
    def read(*command):
        return subprocess.check_output(command, text=True).strip()
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "chip": read("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(read("sysctl", "-n", "hw.memsize")),
        "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")},
        "mlx_gdn_packed": os.environ.get("MLX_GDN_PACKED", "1"),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def prompt_tokens(tokenizer, length):
    unit = tokenizer.encode(CORPUS, add_special_tokens=False)
    return (unit * ((length + len(unit) - 1) // len(unit)))[:length]


def measure(model, tokens, step, output_tokens):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    prompt = mx.array(tokens, dtype=mx.uint32)
    mx.eval(prompt)
    iterator = generate_step(prompt, model, max_tokens=output_tokens, prefill_step_size=step)
    times, generated = [], []
    started = time.perf_counter()
    try:
        for token, _ in iterator:
            times.append(time.perf_counter())
            generated.append(int(token))
    finally:
        iterator.close()
        mx.synchronize()
    ttft = times[0] - started
    decode_seconds = times[-1] - times[0]
    return {
        "prompt_tokens": len(tokens), "prefill_step_size": step,
        "output_tokens": len(generated), "ttft_seconds": ttft,
        "prefill_tps": len(tokens) / ttft,
        "decode_seconds": decode_seconds,
        "decode_tps": (len(generated) - 1) / decode_seconds,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "output_token_ids": generated,
        "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    }


def benchmark(args):
    import mlx.core as mx
    from mlx_lm import load

    path, config = local_model(args.model)
    metadata = environment()
    metadata.update(model_path=str(path), model_config=config, metal=mx.device_info(),
                    config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                    weight_index_sha256=hashlib.sha256((path / "model.safetensors.index.json").read_bytes()).hexdigest(),
                    corpus_sha256=hashlib.sha256(CORPUS.encode()).hexdigest(),
                    measurement="model resident; fresh cache per run; greedy fixed-length raw token input; EOS ignored",
                    timing="TTFT includes first-token sampling; prefill_tps = input tokens / TTFT; decode excludes first token",
                    memory="MLX peak allocated bytes, not total system RAM",
                    seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing to overwrite measurements: {output}")
    print(f"Loading {path}", flush=True)
    start = time.perf_counter()
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    mx.synchronize()
    metadata["load_seconds"] = time.perf_counter() - start
    print(f"Loaded in {metadata['load_seconds']:.2f}s; warming each configuration", flush=True)
    prompts = {n: prompt_tokens(tokenizer, n) for n in args.lengths}
    for n in args.lengths:
        for step in args.steps:
            warm = measure(model, prompts[n], step, 4)
            print(f"Warmed length={n} step={step} TTFT={warm['ttft_seconds']:.3f}s", flush=True)
    schedule = [(n, s, r) for r in range(args.repeats) for n in args.lengths for s in args.steps]
    random.Random(args.seed).shuffle(schedule)
    rows = []
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        handle.flush()
        for n, step, repeat in schedule:
            row = measure(model, prompts[n], step, args.tokens)
            row.update(type="measurement", repeat=repeat)
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"n={n} step={step} repeat={repeat} TTFT={row['ttft_seconds']:.3f}s "
                  f"prefill={row['prefill_tps']:.1f} decode={row['decode_tps']:.1f} tok/s "
                  f"peak={row['peak_memory_gb']:.2f}GB", flush=True)
    summary = summarize(rows)
    report = {"metadata": metadata, "summary": summary, "recommendations": recommend(summary)}
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["recommendations"], indent=2))


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-cache", help="Validate full hybrid-cache snapshots against native MLX-LM")
    verify.add_argument("--model")
    verify.add_argument("--output", default="results/cache-parity.json")
    lookup = commands.add_parser("verify-lookup", help="Measure experimental greedy prompt lookup and cache rollback")
    lookup.add_argument("--model")
    lookup.add_argument("--output", default="results/lookup-parity.json")
    prof = commands.add_parser("profile", help="Synchronized block timing to locate optimization candidates")
    prof.add_argument("--model")
    prof.add_argument("--length", type=positive, default=512)
    prof.add_argument("--step", type=positive, default=2048)
    prof.add_argument("--output", default="results/block-profile.json")
    fusion = commands.add_parser("verify-fusion", help="Screen gate/up Metal QMM packing for numerical and speed regressions")
    fusion.add_argument("--model")
    fusion.add_argument("--output", default="results/fusion.json")
    server = commands.add_parser("serve", help="Localhost-only chat API with exact-prefix caching")
    server.add_argument("--model")
    server.add_argument("--port", type=positive, default=8080)
    server.add_argument("--step", type=positive, default=2048)
    server.add_argument("--context", type=positive, default=32768)
    server.add_argument("--cache-mb", type=positive, default=2000)
    server.add_argument("--memory-mb", type=positive, default=64000,
                        help="MLX allocation guideline and between-step active-memory guard, not an OS hard cap")
    server.add_argument("--allocator-cache-mb", type=positive, default=2000)
    server.add_argument("--lookup", action="store_true", help="Opt in to experimental prompt lookup for copying workloads")
    server.add_argument("--expanded-prefill", action="store_true",
                        help="Experimental prefill-only MLP expansion; needs about 34 GB extra resident memory")
    bench = commands.add_parser("bench", help="Measure fresh-cache text prefill and decode")
    bench.add_argument("--model", help="Local snapshot directory; defaults to cached Qwen3.8-27B-4bit")
    bench.add_argument("--lengths", nargs="+", type=positive, default=[512, 2048, 8192, 32768])
    bench.add_argument("--steps", nargs="+", type=positive, default=[512, 1024, 2048, 4096])
    bench.add_argument("--tokens", type=positive, default=128)
    bench.add_argument("--repeats", type=positive, default=3)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--output", default="results/benchmark.jsonl")
    for command in (bench, verify, lookup, prof, fusion, server):
        command.add_argument("--gdn-kernel", choices=["native", "unpacked"], default="native",
                             help="Select upstream Metal GDN layout; unpacked is an experiment")
    args = parser.parse_args()
    os.environ["MLX_GDN_PACKED"] = "1" if args.gdn_kernel == "native" else "0"
    if args.command == "bench" and args.tokens < 2:
        parser.error("--tokens must be >= 2 to measure decode")
    try:
        if args.command == "verify-cache":
            from .validation import validate
            validate(args)
        elif args.command == "verify-lookup":
            from .validation import validate_lookup
            validate_lookup(args)
        elif args.command == "profile":
            from .profiling import profile
            profile(args)
        elif args.command == "verify-fusion":
            from .fusion import validate_fusion
            validate_fusion(args)
        elif args.command == "serve":
            from .server import serve
            serve(args)
        else:
            benchmark(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
