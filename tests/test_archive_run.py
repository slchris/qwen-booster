import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_non_json_evidence_is_copied_byte_for_byte(tmp_path):
    source = tmp_path / "results"
    source.mkdir()
    (source / "run.log").write_bytes(b"server log line\n")
    (source / "run.jsonl").write_text('{"type": "measurement", "value": 1}\n')
    (source / "run.summary.json").write_text('{"status": "complete"}\n')
    destination = tmp_path / "archive"
    completed = subprocess.run(
        [sys.executable, "scripts/archive_run.py", "--destination", str(destination),
         "--status", "complete", "--note", "test archive",
         str(source / "run.log"), str(source / "run.jsonl"), str(source / "run.summary.json")],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (destination / "run.log").read_bytes() == b"server log line\n"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert set(manifest["artifact_sha256"]) == {"run.log", "run.jsonl", "run.summary.json"}
    assert (destination / "run.summary.json").read_text().startswith("{\n")
