import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("archive_results",
    Path(__file__).parents[1] / "scripts/archive_results.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_hub_snapshot_preserves_actual_repository():
    value = archive.clean({"model_path": "/cache/models--org--model/snapshots/abc"})
    assert value == {"model_repository": "org/model", "model_revision": "abc"}


def test_lmstudio_does_not_invent_revision_or_change_precision():
    value = archive.clean({"model_path": "/user/.lmstudio/models/org/Model-8bit"})
    assert value == {"model_repository": "org/Model-8bit", "model_revision": None}


def test_unknown_directory_has_no_claimed_repository():
    assert archive.clean({"model_path": "/tmp/local-weights"}) == {
        "model_directory_name": "local-weights"}
