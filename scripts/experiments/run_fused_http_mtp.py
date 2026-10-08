"""Serial fused-attention HTTP/MTP/APC comparison; short checks before full 200K."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path.cwd()
PREFIX = 'results/fused-http-mtp-v1'
TARGET = '/Users/chris/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
FLASH = str((ROOT / '.cache/models/qwen3.8-flash-next-4bit-external-ple').resolve())
OVERRIDES = {'PYTHONPATH': 'src', 'MLX_GDN_PACKED': '1', 'APC_CHECKPOINT_ENTRIES': '4', 'APC_MEMORY_RESERVE_GB': str(4_000_000_000 / (1 << 30)), 'APC_DISK_ENABLED': '0', 'APC_CHECKPOINT_INTERVAL_TOKENS': '0'}
ENV = {**os.environ, **OVERRIDES}
MODES = [(mode, TARGET, ['--draft', str(ROOT / '.cache/qwen3.8-mtp-4bit'), '--release-finished-mtp', *flags]) for mode, flags in [('baseline', []), ('fused', ['--fused-prefill'])]]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def status(stage, **extra):
    row = {'stage': stage, 'unix_time': time.time(), 'acceptance_complete': False, **extra}
    Path(PREFIX + '-status.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)


def main():
    check = subprocess.run(['pgrep', '-f', '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py'], capture_output=True, text=True)
    if check.returncode != 1:
        raise RuntimeError('Existing model process or failed process audit')
    if Path(PREFIX + '-config.json').exists():
        raise RuntimeError('Previous run exists')
    audit = json.loads(Path('results/reasoning-diverse-200k-token-audit-v1.json').read_text())
    for model in audit['audits']:
        assert model['suite_sha256'] == digest('results/reasoning-diverse-200k-suite-v1.jsonl')
        assert len(model['cases']) == 3
        assert all(c['fits_native_context'] and c['meets_minimum_input'] for c in model['cases'])
    commands = {}
    for mode, model, flags in MODES:
        commands[mode] = ['.cache/vlm-env/bin/python', '-m', 'qwen_booster.vlm_server', '--model', model, '--context', '262144', '--port', '18083', '--cache-mb', '24000', '--memory-mb', '110000', '--allocator-cache-mb', '2000', '--token-timeout', '7200', *flags]
    files = list(Path('src/qwen_booster').glob('*.py')) + list(Path('scripts').glob('*.py')) + [Path(__file__)] + list(Path('results').glob('reasoning-diverse-*-v1.json*'))
    hashes = {str(p): digest(p) for p in files}
    Path(PREFIX + '-config.json').write_text(json.dumps({'commands': commands, 'environment_overrides': OVERRIDES, 'source_sha256': hashes, 'scope': 'Same three original tasks, both baseline and fused with MTP and full APC. Complete both short checks before 200K. Fresh server per mode/tier, one cold/warm pair per task; no stable speedup or broad quality acceptance. No input reduction.'}, indent=2) + '\n')
    for tier, (mode, model, _) in [(tier, mode) for tier in ['short', '200k'] for mode in MODES]:
        stem = PREFIX + '-' + tier + '-' + mode
        with Path(stem + '-server.log').open('x') as log:
            server = subprocess.Popen(commands[mode], env=ENV, stdout=log, stderr=subprocess.STDOUT)
            monitor = subprocess.Popen(['.venv/bin/python', 'scripts/monitor_memory.py', '--pid', str(server.pid), '--output', stem + '-memory.jsonl'])
            try:
                status('loading', mode=mode, pid=server.pid)
                deadline = time.monotonic() + 1200
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Model server exited')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18083/health', timeout=5) as response:
                            if json.load(response).get('loaded_model') == model:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError('Model readiness timeout')
                    time.sleep(5)
                for tier in [tier]:
                    suite = f'results/reasoning-diverse-{tier}-suite-v1.jsonl'
                    for case in map(json.loads, Path(suite).read_text().splitlines()):
                        assert all(digest(p) == h for p, h in hashes.items()), 'Experiment sources changed'
                        output = stem + '-' + case['id'] + '.jsonl'
                        status('evaluating', mode=mode, tier=tier, case_id=case['id'], pid=server.pid, output=output)
                        subprocess.run(['.venv/bin/python', 'scripts/evaluate_task_replay.py', '--suite', suite, '--case-id', case['id'], '--model', model, '--endpoint', 'http://127.0.0.1:18083/v1', '--backend', 'vlm', '--output', output], env=ENV, check=True)
                        rows = [json.loads(line) for line in Path(output).read_text().splitlines()]
                        assert len(rows) == 4 and rows[-1]['type'] == 'comparison'
                        for row, phase in zip(rows[1:3], ['cold', 'warm']):
                            r = row['result']; t = r['server_timings']; n = r['usage']['prompt_tokens']
                            assert r['complete'] and r['finish_reason'] == 'stop' and not r['server_errors']
                            assert n + case['max_tokens'] <= 262144
                            if tier == '200k':
                                entry = next(a for a in audit['audits'] if Path(a['model_path']).resolve() == Path(model).resolve())
                                expected = next(c['actual_prompt_tokens'] for c in entry['cases'] if c['case_id'] == case['id'])
                                assert n == expected and n >= 204800
                            assert t['cache_n'] == (0 if phase == 'cold' else n - 1), 'Cache regime mismatch'
                        if tier == 'short':
                            assert rows[-1]['quality_passed'] and rows[-1]['answer_parity'] and rows[-1]['reasoning_parity'], 'Short integration quality/parity failed'
                        if mode == 'fused':
                            assert 'FUSED_PREFILL_ACTIVE' in Path(stem + '-server.log').read_text(), 'Fused dispatch not observed'
                        status('pair_recorded', mode=mode, tier=tier, case_id=case['id'], comparison=rows[-1])
                assert all(digest(p) == h for p, h in hashes.items()), 'Experiment sources changed'
            finally:
                if server.poll() is None:
                    server.terminate()
                server.wait(timeout=120)
                monitor.wait(timeout=30)
    status('observations_finished_pending_review')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_pending_review', error=str(error))
        raise
