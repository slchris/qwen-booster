"""200K full-history regression for fused prefill, MTP and complete prefix caching."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

PREFIX = 'results/fused-mtp-conversation-200k-v1'
MODEL = '/Users/chris/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
OVERRIDES = {'PYTHONPATH': 'src', 'MLX_GDN_PACKED': '1', 'APC_CHECKPOINT_ENTRIES': '4', 'APC_DISK_ENABLED': '0', 'APC_CHECKPOINT_INTERVAL_TOKENS': '0', 'APC_MEMORY_RESERVE_GB': str(4_000_000_000 / (1 << 30))}
ENV = {**os.environ, **OVERRIDES}
SUITE = 'results/context-200k-v1-suite.jsonl'
SHORT_SUITE = 'benchmarks/mtp-long-quality-v1/long-task-quality-suite.jsonl'


def status(stage, **extra):
    data = {'stage': stage, 'unix_time': time.time(), **extra}
    Path(PREFIX + '-status.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)


def main():
    live = subprocess.run(['pgrep', '-f', '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py|[b]enchmark_fused_prefill.py'], capture_output=True, text=True)
    if live.returncode != 1:
        raise RuntimeError('Existing GPU process or failed process audit')
    if list(Path('results').glob('fused-mtp-conversation-200k-v1*')):
        raise RuntimeError('Prior artifacts exist; refusing to overwrite')
    gate = json.loads(Path('results/fused-http-mtp-v1-review.json').read_text())
    assert gate['reviewed_requests'] == gate['content_correct'] == gate['bare_json_correct'] == 24
    for path, expected in gate['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    common = ['.cache/vlm-env/bin/python', '-m', 'qwen_booster.vlm_server', '--model', MODEL, '--context', '262144', '--port', '18083', '--cache-mb', '24000', '--memory-mb', '110000', '--allocator-cache-mb', '2000', '--token-timeout', '7200']
    sources = list(Path('src/qwen_booster').glob('*.py')) + list(Path('scripts').glob('*.py')) + [Path(__file__), Path(SUITE), Path(SHORT_SUITE)]
    config = {'commands': {'candidate': common + ['--draft', str(Path('.cache/qwen3.8-mtp-4bit').resolve()), '--release-finished-mtp', '--fused-prefill']}, 'environment_overrides': OVERRIDES, 'suite': SUITE, 'short_suite': SHORT_SUITE, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'prior_code_gate_evidence': gate['source_sha256'], 'scope': 'Same unchanged 204801-token fixture as prior MTP/Flash conversation. Fused prefill + MTP + full APC, ten real turns and final identical cache-cleared replay. Four short turns first. No broad quality, native-boundary or controlled historical speedup claim.'}
    with Path(PREFIX + '-config.json').open('x') as f:
        json.dump(config, f, indent=2)
    for mode, command in config['commands'].items():
        with Path(PREFIX + '-' + mode + '-server.log').open('x') as log:
            server = subprocess.Popen(command, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            monitor = None
            try:
                monitor = subprocess.Popen(['.venv/bin/python', 'scripts/monitor_memory.py', '--pid', str(server.pid), '--output', PREFIX + '-' + mode + '-memory.jsonl'])
                status('loading', mode=mode, pid=server.pid)
                deadline = time.monotonic() + 1200
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Server exited during loading')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18083/health', timeout=5) as response:
                            if json.load(response).get('loaded_model') == MODEL:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Readiness deadline')
                    time.sleep(5)
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in config['source_sha256'].items())
                status('evaluating', mode=mode, pid=server.pid)
                checks = [
                    ['scripts/evaluate_sessions.py', '--backend', 'vlm', '--turns', '4', '--cold-every', '1',
                     '--prefix-case', SHORT_SUITE, '--prefix-id', 'long-code-8192-thinking-False',
                     '--output', PREFIX + '-short.jsonl'],
                    ['scripts/evaluate_200k.py', '--backend', 'vlm', '--stop-on-cache-loss',
                     '--suite', SUITE, '--output', PREFIX + '-200k.jsonl'],
                ]
                for arguments in checks:
                    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in config['source_sha256'].items())
                    status('evaluating', script=arguments[0], pid=server.pid)
                    result = subprocess.run(['.venv/bin/python', *arguments, '--model', MODEL,
                                             '--endpoint', 'http://127.0.0.1:18083/v1'], env=ENV)
                    if result.returncode:
                        raise RuntimeError('Incomplete conversation; preserve evidence and inspect before proceeding')
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in config['source_sha256'].items())
                assert 'FUSED_PREFILL_ACTIVE' in Path(PREFIX + '-' + mode + '-server.log').read_text()
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
