"""Full-history admission diagnostic with byte telemetry; retain incorrect answers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

PREFIX = 'results/borrow-27b-261k-admission-v1'
SUITE = 'results/native-27b-failure-replay-v1-suite.jsonl'
AUDIT = 'results/vlm-27b-261k-replay-token-audit-v1.json'
MODEL = '/Users/chris/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **extra):
    row = dict(stage=stage, unix_time=time.time(), **extra)
    Path(PREFIX + '-status.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)


def main():
    check = subprocess.run(['pgrep', '-f', '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py|[b]enchmark_fused_prefill.py'], capture_output=True, text=True)
    if check.returncode != 1:
        raise RuntimeError('Existing GPU process or failed process audit')
    if list(Path('results').glob('borrow-27b-261k-admission-v1*')):
        raise RuntimeError('Prior outputs exist')
    gate = json.loads(Path('results/borrow-27b-http-short-v1-review.json').read_text())
    assert gate['responses_quality_passed'] == 12 and gate['cold_warm_pairs_passed'] == 6
    assert all(all(c['between_modes_parity'].values()) for c in gate['cases'])
    audit = json.loads(Path(AUDIT).read_text())['audits'][0]
    assert audit['suite_sha256'] == digest(SUITE)
    counts = audit['cases'][0]
    assert counts['actual_prompt_tokens'] == 261265
    assert counts['output_budget_tokens'] == 32 and counts['fits_native_context']
    case = json.loads(Path(SUITE).read_text().splitlines()[0])
    command = ['.cache/vlm-env/bin/python', 'scripts/trace_apc_server.py', '--model', MODEL,
               '--context', '262144', '--port', '18083', '--cache-mb', '48000',
               '--memory-mb', '110000', '--allocator-cache-mb', '2000', '--token-timeout', '7200', '--draft', str(Path('.cache/qwen3.8-mtp-4bit').resolve()), '--release-finished-mtp', '--fused-prefill', '--borrow-27b-checkpoint-row']
    overrides = {'PYTHONPATH': 'src', 'MLX_GDN_PACKED': '1', 'APC_CHECKPOINT_ENTRIES': '4',
                 'APC_DISK_ENABLED': '0', 'APC_CHECKPOINT_INTERVAL_TOKENS': '0', 'APC_MEMORY_RESERVE_GB': str(4_000_000_000 / (1 << 30))}
    env = {**os.environ, **overrides}
    sources = list(Path('src/qwen_booster').glob('*.py')) + [Path(__file__), Path(SUITE), Path(AUDIT), Path('results/borrow-27b-http-short-v1-review.json')]
    sources += list(Path('scripts').glob('*.py'))
    hashes = {str(p): digest(p) for p in sources}
    config = {'command': command, 'environment_overrides': overrides, 'source_sha256': hashes,
              'scope': 'Same full 261265-token historical failure, Q4 greedy no-thinking, cold then identical warm on VLM. Fused prefill + MTP + full APC with final-only checkpoints, no disk. 48GB cache budget. This is a combined-stack quality diagnostic, not an isolated speed comparison to historical baseline. Wrong answers are retained. Candidate borrowed singleton 27B checkpoint normalization enabled; all original admission guards unchanged. Read-only _make_room byte tracing is enabled, so this is not a timing benchmark. Skip replay if no RAM snapshot remains.'}
    with Path(PREFIX + '-config.json').open('x') as out:
        json.dump(config, out, indent=2)
    with Path(PREFIX + '-server.log').open('x') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = None
        try:
            monitor = subprocess.Popen(['.venv/bin/python', 'scripts/monitor_memory.py', '--pid', str(server.pid), '--output', PREFIX + '-memory.jsonl'])
            status('loading', pid=server.pid)
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Server exited before readiness')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18083/health', timeout=5) as response:
                        if json.load(response).get('loaded_model') == MODEL:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Readiness timeout')
                time.sleep(5)
            status('evaluating', pid=server.pid)
            subprocess.run(['.venv/bin/python', 'scripts/evaluate_admission_replay.py', '--suite', SUITE,
                            '--output', PREFIX + '.jsonl', '--endpoint',
                            'http://127.0.0.1:18083/v1', '--model', MODEL], env=env, check=True)
            assert all(digest(p) == h for p, h in hashes.items()), 'Sources changed during test'
            assert 'BORROW_27B_CHECKPOINT_ROW' in Path(PREFIX + '-server.log').read_text()
            assert 'FUSED_PREFILL_ACTIVE' in Path(PREFIX + '-server.log').read_text()
        finally:
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=120)
            if monitor:
                monitor.wait(timeout=30)
    status('finished_pending_review', acceptance_complete=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_pending_review', error=str(error))
        raise
