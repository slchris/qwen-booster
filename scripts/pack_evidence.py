"""Pack an evidence directory into a deterministic, checksummed tarball.

The source repository does not track ``benchmarks/``. Release bundles are
built from a maintained working copy and attached to a release. Deterministic
output (sorted paths, zeroed ownership and mtimes, gzip mtime 0) keeps the
SHA-256 stable for identical content so downloads can be verified.

Usage:
  python scripts/pack_evidence.py --source benchmarks \
      --output dist/qwen-booster-evidence-v0.1.0.tar.gz
"""

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_version():
    try:
        import tomllib
        with (ROOT / "pyproject.toml").open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]
    except Exception:
        return "0.0.0"


def _normalized(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def pack(source, output):
    """Write ``output`` and its ``.sha256`` sidecar; refuse to overwrite."""
    source = Path(source)
    output = Path(output)
    if not source.is_dir():
        raise FileNotFoundError(f"evidence source not found: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"no files under {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0,
                           compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    info = _normalized(archive.gettarinfo(str(path)))
                    info.name = str(Path(source.name) / path.relative_to(source))
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{digest.hexdigest()}  {output.name}\n")
    return {"files": len(files), "bytes": output.stat().st_size,
            "sha256": digest.hexdigest(), "output": str(output),
            "checksum_file": str(sidecar)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="benchmarks")
    parser.add_argument("--output",
                        default=f"dist/qwen-booster-evidence-v{project_version()}.tar.gz")
    args = parser.parse_args()
    report = pack(args.source, args.output)
    print(f"packed {report['files']} files into {report['output']} "
          f"({report['bytes'] / 1e6:.1f} MB)\nsha256 {report['sha256']}")


if __name__ == "__main__":
    main()
