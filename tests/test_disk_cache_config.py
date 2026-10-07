from pathlib import Path
import os
import subprocess
import sys

import pytest

from qwen_booster.vlm_server import disk_cache_environment


def test_disk_cache_default_disabled_and_explicit_capacity(tmp_path):
    assert disk_cache_environment(None, 0, 0) == {'APC_DISK_ENABLED': '0'}
    env = disk_cache_environment(str(tmp_path / 'snapshots'), 12000, 12000)
    assert env['APC_DISK_ENABLED'] == '1'
    assert Path(env['APC_DISK_PATH']) == tmp_path / 'snapshots'
    assert float(env['APC_DISK_MAX_GB']) * (1 << 30) == 12_000_000_000
    assert not (tmp_path / 'snapshots').exists()


@pytest.mark.parametrize('directory,budget,memory', [
    (None, 1, 1), ('cache', 0, 1), ('cache', -1, 1), ('cache', 1, 0)])
def test_incomplete_or_disabled_cache_configuration_rejected(directory, budget, memory):
    with pytest.raises(ValueError):
        disk_cache_environment(directory, budget, memory)


def test_file_cannot_be_used_as_cache_directory(tmp_path):
    path = tmp_path / 'file'
    path.write_text('existing data')
    with pytest.raises(ValueError, match='directory'):
        disk_cache_environment(str(path), 1, 1)
    assert path.read_text() == 'existing data'


def test_module_entrypoint_rejects_incomplete_disk_settings_before_model_load():
    process = subprocess.run(
        [sys.executable, '-m', 'qwen_booster.vlm_server', '--model', 'not-loaded',
         '--disk-cache-mb', '12000'], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')})
    assert process.returncode == 2
    assert 'disk cache requires both' in process.stderr
    assert 'Traceback' not in process.stderr
