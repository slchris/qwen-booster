import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'pack_evidence', Path(__file__).parents[1] / 'scripts/pack_evidence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_tree(tmp_path):
    source = tmp_path / "benchmarks"
    (source / "one").mkdir(parents=True)
    (source / "one" / "manifest.json").write_text('{"status": "complete"}\n')
    (source / "two.log").write_text("server log\n")
    return source


def test_pack_is_deterministic_and_checksummed(tmp_path):
    source = make_tree(tmp_path)
    first = tmp_path / "a.tar.gz"
    second = tmp_path / "b.tar.gz"
    report_a = module.pack(source, first)
    report_b = module.pack(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert report_a["sha256"] == report_b["sha256"]
    assert report_a["files"] == 2
    sidecar = first.with_name(first.name + ".sha256").read_text()
    assert sidecar == f"{report_a['sha256']}  {first.name}\n"


def test_pack_refuses_overwrite_and_missing_source(tmp_path):
    source = make_tree(tmp_path)
    output = tmp_path / "evidence.tar.gz"
    module.pack(source, output)
    with pytest.raises(FileExistsError):
        module.pack(source, output)
    with pytest.raises(FileNotFoundError):
        module.pack(tmp_path / "missing", tmp_path / "other.tar.gz")


def test_pack_extracts_back_to_benchmarks_prefix(tmp_path):
    import tarfile
    source = make_tree(tmp_path)
    output = tmp_path / "evidence.tar.gz"
    module.pack(source, output)
    extract = tmp_path / "extract"
    extract.mkdir()
    with tarfile.open(output) as archive:
        names = sorted(archive.getnames())
        archive.extractall(extract, filter="data")
    assert names == ["benchmarks/one/manifest.json", "benchmarks/two.log"]
    assert (extract / "benchmarks" / "one" / "manifest.json").is_file()
