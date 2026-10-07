import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'snapshot_summary', Path(__file__).parents[1] / 'scripts/summarize_snapshot_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(phase, active):
    return 'INFO APC_SNAPSHOT_DIAGNOSTIC ' + json.dumps({
        'phase': phase, 'active_bytes': active, 'allocator_cache_bytes': 0,
        'resident_bytes': 0})


def test_failed_admission_does_not_invent_clone_measurement():
    report = module.summarize('\n'.join([
        event('checkpoint.before', 100), event('extract.before', 100),
        event('extract.after', 150),
        'INFO APC_MEMORY_DIAGNOSTIC {"admitted": false}',
        event('checkpoint.after', 100)]))
    assert report['parse_errors'] == []
    row = report['checkpoints'][0]
    assert row['allocation_deltas']['extract']['active_bytes'] == 50
    assert 'clone' not in row['allocation_deltas']
    assert row['admission'] == [{'admitted': False}]
    assert not report['acceptance_complete']


def test_incomplete_checkpoint_is_reported():
    report = module.summarize(event('checkpoint.before', 100))
    assert report['parse_errors'] == ['final checkpoint incomplete']
