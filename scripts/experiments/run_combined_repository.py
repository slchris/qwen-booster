"""Current 27B combined stack versus MTP control on unchanged full-source tasks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path.cwd()
PREFIX = 'results/combined-27b-repository-v1'
SUITE = Path('results/repository-behavior-200k-suite-v1.jsonl')
TARGET = '/Users/chris/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
OVERRIDES = {'PYTHONPATH': 'src', 'MLX_GDN_PACKED': '1',
    'APC_CHECKPOINT_ENTRIES': '4', 'APC_MEMORY_RESERVE_GB': str(4_000_000_000 / (1 << 30)),
    'APC_DISK_ENABLED': '0', 'APC_CHECKPOINT_INTERVAL_TOKENS': '0', 'APC_TRACE': '0'}
FLAGS = {'control': [], 'combined': ['--fused-prefill', '--borrow-27b-checkpoint-row',
    '--credit-27b-live-warm-cache', '--recheck-27b-cache-admission']}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **extra):
    row = {'stage': stage, 'unix_time': time.time(), 'acceptance_complete': False, **extra}
    Path(PREFIX + '-status.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)


def main():
    check = subprocess.run(['pgrep', '-f', '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py|[b]enchmark_fused_prefill.py|[p]rofile_long_attention.py|[s]creen_head_group_attention.py'], capture_output=True, text=True)
    if check.returncode != 1:
        raise RuntimeError('Existing model process or failed process audit')
    if list(Path('results').glob(Path(PREFIX).name + '-*')):
        raise RuntimeError('Output prefix already exists')
    gate_path = Path('results/recheck-27b-switching-conversation-200k-v1-review.json')
    gate = json.loads(gate_path.read_text())
    assert gate['bounded_switching_passed'] and gate['full_history_retokenization_passed']
    assert gate['recheck']['recovered'] > 0 and gate['recheck']['sampled_guards_passed']
    assert all(digest(p) == h for p, h in gate['source_sha256'].items())
    previous = json.loads(Path('results/recheck-27b-switching-conversation-200k-v1-config.json').read_text())
    assert all(digest(p) == h for p, h in previous['source_sha256'].items() if p.startswith('src/'))
    assert 'Application shutdown complete.' in Path('results/recheck-27b-switching-conversation-200k-v1-27b-combined-server.log').read_text()
    audit_path = Path('results/repository-behavior-200k-token-audit-v1.json')
    audit = next(a for a in json.loads(audit_path.read_text())['audits'] if a['model_id'] == 'qwen3.8-27b')
    assert digest(SUITE) == audit['suite_sha256']
    assert all(digest(Path(TARGET) / p) == h for p, h in audit['model_file_sha256'].items())
    assert all(c['fits_native_context'] and c['meets_minimum_input'] for c in audit['cases'])
    cases = [json.loads(line) for line in SUITE.read_text().splitlines()]
    assert len(cases) == 2
    sequence = [(mode, case) for i, case in enumerate(cases)
                for mode in (['control', 'combined'] if i == 0 else ['combined', 'control'])]
    commands = {mode: ['.cache/vlm-env/bin/python', '-m', 'qwen_booster.vlm_server',
        '--model', TARGET, '--context', '262144', '--port', '18083', '--cache-mb', '48000',
        '--memory-mb', '110000', '--allocator-cache-mb', '2000', '--token-timeout', '7200',
        '--draft', str(ROOT / '.cache/qwen3.8-mtp-4bit'), '--release-finished-mtp', *flags]
        for mode, flags in FLAGS.items()}
    files = list(Path('src/qwen_booster').glob('*.py')) + list(Path('scripts').glob('*.py'))
    files += [SUITE, audit_path, gate_path, Path('results/repository-behavior-cpu-oracles-v1.json'),
              Path('results/repository-behavior-oracle-builder-v1.py'),
              Path('results/combined-27b-repository-next-stage-plan-v1.json')]
    hashes = {str(p): digest(p) for p in files}
    env = {**os.environ, **OVERRIDES}
    # Disable inherited diagnostic tracing as well as ordinary APC traces.
    for name in ('APC_MEMORY_DIAGNOSTIC', 'APC_MEMORY_TRACE'):
        env.pop(name, None)
    Path(PREFIX + '-config.json').write_text(json.dumps({'commands': commands,
        'environment_overrides': OVERRIDES, 'source_sha256': hashes,
        'sequence': [{'mode': mode, 'case_id': case['id']} for mode, case in sequence],
        'scope': 'Two unchanged full-source >=204800-token tasks, two configurations, cold/warm pair each. Fresh server per pair; reverse mode order between tasks. One observation per cell, not P95 or stable causal speedup evidence. Content and format failures retained separately.'}, indent=2) + '\n')
    def verify():
        assert all(digest(p) == h for p, h in hashes.items()), 'Experiment sources changed'
    for index, (mode, case) in enumerate(sequence):
        verify()
        stem = PREFIX + '-' + mode + '-' + case['id']
        with Path(stem + '-server.log').open('x') as log:
            server = subprocess.Popen(commands[mode], env=env, stdout=log, stderr=subprocess.STDOUT)
            monitor = subprocess.Popen(['.venv/bin/python', 'scripts/monitor_memory.py',
                '--pid', str(server.pid), '--output', stem + '-memory.jsonl'])
            try:
                status('loading', index=index, mode=mode, case_id=case['id'], pid=server.pid)
                deadline = time.monotonic() + 1200
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Server exited during startup')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18083/health', timeout=5) as response:
                            if json.load(response).get('loaded_model') == TARGET:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError('Model readiness timeout')
                    time.sleep(5)
                status('evaluating', index=index, mode=mode, case_id=case['id'], pid=server.pid)
                subprocess.run(['.venv/bin/python', 'scripts/evaluate_task_replay.py',
                    '--suite', str(SUITE), '--case-id', case['id'], '--model', TARGET,
                    '--endpoint', 'http://127.0.0.1:18083/v1', '--backend', 'vlm',
                    '--output', stem + '.jsonl'], env=env, check=True)
                rows = [json.loads(line) for line in Path(stem + '.jsonl').read_text().splitlines()]
                assert len(rows) == 4 and rows[-1]['type'] == 'comparison'
                expected = next(c['actual_prompt_tokens'] for c in audit['cases'] if c['case_id'] == case['id'])
                for row, phase in zip(rows[1:3], ['cold', 'warm']):
                    r = row['result']; t = r['server_timings']; n = r['usage']['prompt_tokens']
                    assert r['complete'] and r['finish_reason'] == 'stop' and not r['server_errors']
                    assert n == expected and n >= 204800 and n + case['max_tokens'] <= 262144
                    assert t['cache_n'] == (0 if phase == 'cold' else n - 1), 'Cache regime mismatch'
                verify()
                status('pair_recorded', index=index, mode=mode, case_id=case['id'], comparison=rows[-1])
            finally:
                if server.poll() is None:
                    server.terminate()
                server.wait(timeout=120)
                monitor.wait(timeout=30)
    verify()
    status('observations_finished_pending_review')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_pending_review', error=str(error))
        raise
