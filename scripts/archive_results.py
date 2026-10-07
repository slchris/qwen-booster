"""Archive the completed reference run without absolute local model paths."""

import json
import hashlib
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        result = {k: clean(v) for k, v in value.items() if k != "model_path"}
        if "model_path" in value:
            path = Path(value["model_path"])
            parts = path.parts
            hub = next((part for part in parts if part.startswith("models--")), None)
            if hub and path.parent.name == "snapshots":
                result["model_repository"] = hub.removeprefix("models--").replace("--", "/", 1)
                result["model_revision"] = path.name
            elif ".lmstudio" in parts and "models" in parts:
                result["model_repository"] = f"{path.parent.name}/{path.name}"
                result["model_revision"] = None
            else:
                result["model_directory_name"] = path.name
                # A local directory name is not evidence of a repository revision.
        return result
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    source = Path("results")
    destination = Path("benchmarks/m3-max-qwen3.8-27b-4bit")
    names = ["baseline-tuning.jsonl", "baseline-tuning.summary.json", "tuning-confirmation.jsonl",
             "tuning-confirmation.summary.json", "long-context.jsonl", "long-context.summary.json",
             "cache-parity.json", "lookup-parity.json", "fusion-final.json", "gdn-layout.json", "api-smoke.json"]
    for name in names:
        if not (source / name).is_file():
            raise SystemExit(f"Required evidence missing: {name}")
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    for name in names:
        raw = (source / name).read_text()
        if name.endswith("jsonl"):
            content = "".join(json.dumps(clean(json.loads(line)), ensure_ascii=False) + "\n"
                              for line in raw.splitlines())
        else:
            content = json.dumps(clean(json.loads(raw)), ensure_ascii=False, indent=2) + "\n"
        (destination / name).write_text(content)
    profile = json.loads((source / "block-profile.json").read_text())
    profile.pop("records")
    (destination / "block-profile.summary.json").write_text(json.dumps(clean(profile), indent=2) + "\n")
    metadata = clean(json.loads((source / "baseline-tuning.jsonl").read_text().splitlines()[0]))
    manifest = {
        "model_repository": metadata["model_repository"],
        "model_revision": metadata["model_revision"],
        "versions": metadata["versions"],
        "chip": metadata["chip"],
        "memory_bytes": metadata["memory_bytes"],
        "limitations": [
            "First matrix had temporal performance drift; do not infer cross-run speedups.",
            "Lookup gain is from one copying case with warm cache, fixed 128-token greedy generation.",
            "Cache TTFT samples exclude tokenization and HTTP; correctness covers six cases.",
            "No comparison against BF16 quality, MTP, vision, or llama.cpp/GGUF in this run.",
        ],
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob("*.json*"))},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived reference measurements in {destination}")


if __name__ == "__main__":
    main()
