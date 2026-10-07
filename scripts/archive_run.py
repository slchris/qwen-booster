"""Archive explicit completed/partial evidence with provenance and honest run status."""

import argparse
import hashlib
import json
from pathlib import Path

from archive_results import clean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--status", choices=["partial", "complete"], required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    sources = [Path(name) for name in args.files]
    if len({p.name for p in sources}) != len(sources):
        parser.error("source basenames must be unique")
    # Validate all evidence before creating the destination.
    contents = {}
    for path in sources:
        if path.suffix == ".jsonl":
            data = [clean(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
            contents[path.name] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        else:
            contents[path.name] = json.dumps(clean(json.loads(path.read_text())), ensure_ascii=False, indent=2) + "\n"
    dest = Path(args.destination)
    dest.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        (dest / name).write_text(content)
    source_code = sorted(Path("src/qwen_booster").glob("*.py")) + sorted(Path("scripts").glob("*.py"))
    manifest = {"status": args.status, "note": args.note,
                "artifact_sha256": {name: digest(dest / name) for name in contents},
                "source_sha256_at_archive_time": {str(p): digest(p) for p in source_code},
                "warning": "Archive-time source hashes do not establish the code used by older runs; use run metadata."}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
