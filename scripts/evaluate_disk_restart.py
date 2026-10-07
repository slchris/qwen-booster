"""Flash-Next complete-model checkpoint recovery across a server restart.

Uses a fresh bounded disk directory and a deliberately tiny memory cache so the
second request cannot be mistaken for an in-memory hit. Run GPU jobs serially.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from evaluate_api import evaluate
from qwen_booster.grading import grade_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--disk-dir', required=True)
    parser.add_argument('--disk-mb', type=int, required=True)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()
    if args.disk_mb <= 0:
        parser.error('disk budget must be positive')
    root = Path(args.disk_dir).resolve()
    prefix = Path(args.output_prefix)
    raw = Path(args.suite).read_bytes()
    matches = [c for c in map(json.loads, raw.splitlines()) if c['id'] == args.case_id]
    if len(matches) != 1 or not matches[0].get('grading'):
        parser.error('select exactly one graded fixture')
    case = matches[0]
    live = subprocess.run(['pgrep', '-f',
        '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py'],
        capture_output=True, text=True)
    if live.returncode != 1:
        raise RuntimeError('Existing model process or failed process audit; refusing GPU launch')
    root.mkdir(parents=True, exist_ok=False)
    model = str(Path(args.model).resolve())
    env = {**os.environ, 'PYTHONPATH': 'src', 'APC_CHECKPOINT_ENTRIES': '1',
           'APC_MEMORY_RESERVE_GB': str(4_000_000_000 / (1 << 30))}
    command = ['.cache/vlm-env/bin/python', '-m', 'qwen_booster.vlm_server',
               '--model', model, '--context', '262144', '--port', '18081',
               '--cache-mb', '1', '--memory-mb', '110000', '--allocator-cache-mb', '2000',
               '--disk-cache-dir', str(root), '--disk-cache-mb', str(args.disk_mb),
               '--credit-live-warm-cache', '--borrow-single-checkpoint-row']
    endpoint = 'http://127.0.0.1:18081'

    def save(suffix, value):
        with Path(str(prefix) + suffix).open('x') as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write('\n')

    def status(stage, **extra):
        Path(str(prefix) + '-status.json').write_text(json.dumps(
            {'stage': stage, 'unix_time': time.time(), **extra}, indent=2) + '\n')
        print(stage, extra, flush=True)

    sources = list(Path('src/qwen_booster').glob('*.py')) + list(Path('scripts').glob('*.py'))
    save('-config.json', {'command': command, 'case_id': args.case_id,
                         'suite_sha256': hashlib.sha256(raw).hexdigest(),
                         'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                         'environment_overrides': {k: env[k] for k in ('APC_CHECKPOINT_ENTRIES', 'APC_MEMORY_RESERVE_GB')},
                         'scope': __doc__})
    observations = []
    try:
        for phase in ('cold', 'disk_restore'):
            with Path(str(prefix) + '-' + phase + '-server.log').open('x') as log:
                server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                monitor = None
                try:
                    monitor = subprocess.Popen(['.venv/bin/python', 'scripts/monitor_memory.py',
                        '--pid', str(server.pid), '--output', str(prefix) + '-' + phase + '-memory.jsonl'])
                    status('loading', phase=phase, pid=server.pid)
                    deadline = time.monotonic() + 1200
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError('server exited during load')
                        try:
                            with urllib.request.urlopen(endpoint + '/health', timeout=5) as response:
                                health = json.load(response)
                            if health.get('loaded_model') == model:
                                break
                        except (OSError, TimeoutError):
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError('model readiness timed out')
                        time.sleep(5)
                    status('evaluating', phase=phase, pid=server.pid)
                    result = evaluate(endpoint + '/v1', case, 7200, model=model, include_usage=True)
                    with urllib.request.urlopen(endpoint + '/v1/cache/stats', timeout=30) as response:
                        stats = json.load(response)
                    observation = {'phase': phase, 'pid': server.pid, 'result': result, 'cache_stats': stats}
                    observations.append(observation)
                    save('-' + phase + '.json', observation)
                    if not result['complete'] or result['finish_reason'] != 'stop':
                        raise RuntimeError('request did not finish normally')
                finally:
                    if server.poll() is None:
                        server.terminate()
                    server.wait(timeout=120)
                    if monitor is not None:
                        monitor.wait(timeout=30)
            files = list(root.rglob('*.safetensors'))
            if not files:
                raise RuntimeError('no persisted snapshot after server shutdown')
        cold, warm = [o['result'] for o in observations]
        stats = observations[1]['cache_stats']
        correct = all(grade_answer(r['text'], case['grading']) is True for r in (cold, warm))
        parity = all(cold[k] == warm[k] for k in ('text', 'reasoning'))
        expected = case['prompt_tokens']
        counts = all(r['usage']['prompt_tokens'] == expected for r in (cold, warm))
        actual_disk = (cold['server_timings']['cache_n'] == 0
                       and warm['server_timings']['cache_n'] >= expected - 1
                       and stats.get('disk_hits', 0) > 0
                       and stats.get('exact_resident_bytes', 0) == 0)
        report = {'bounded_disk_restart_gate_passed': correct and parity and counts and actual_disk,
                  'quality_passed': correct, 'body_and_reasoning_equal': parity,
                  'actual_token_counts_match': counts, 'disk_reuse_proven': actual_disk,
                  'input_tokens': expected, 'cold_seconds': cold['seconds'], 'disk_restore_seconds': warm['seconds'],
                  'disk_files': len(files), 'disk_bytes': sum(p.stat().st_size for p in files),
                  'acceptance_complete': False, 'scope': 'One fixture and restart, single observations; no general quality or speedup claim.'}
        save('-review.json', report)
        status('reviewed', **report)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
