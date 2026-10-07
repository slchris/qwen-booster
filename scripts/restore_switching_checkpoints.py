"""Restore existing two-document snapshots in separate fresh server processes.

Cold references are historical identical requests, not a randomized speed trial.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from evaluate_api import evaluate
from audit_workload_tokens import count_case
from qwen_booster.grading import grade_answer, grade_json_format
from run_document_switching import ROOT, SUITE, prepare
from transformers import AutoTokenizer


def main():
    os.chdir(ROOT)
    original = ROOT / 'results/document-switching-disk-27b-200k-v1'
    prefix = ROOT / 'results/document-switching-27b-restart-v1'
    if list(prefix.parent.glob(prefix.name + '-*')):
        raise RuntimeError('Refusing to overwrite observations')
    config = json.loads(Path(str(original) + '-config.json').read_text())
    for name, expected in config['source_sha256'].items():
        if '/src/' in name and hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Runtime changed since snapshot creation: ' + name)
    runs, _ = prepare()
    run = next(r for r in runs if r['mode'] == '27b-normalized')
    assert run == config['run']
    audit = subprocess.run(['pgrep', '-f', '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py|[b]enchmark_fused_prefill.py|[p]rofile_long_attention.py|[s]creen_head_group_attention.py'], capture_output=True, text=True)
    if audit.returncode != 1:
        raise RuntimeError('GPU process audit failed: ' + audit.stdout)
    cases = [json.loads(line) for line in SUITE.read_text().splitlines()]
    cold = {r['name']: r for r in map(json.loads, Path(str(original) + '-responses.jsonl').read_text().splitlines()) if r.get('type') == 'measurement'}
    tokenizer = AutoTokenizer.from_pretrained(run['model'], local_files_only=True)
    command = run['command'] + ['--disk-cache-dir', config['disk_directory'], '--disk-cache-mb', str(config['disk_budget_mb'])]
    files = list(Path(config['disk_directory']).rglob('*.safetensors'))
    assert files
    def save(suffix, data):
        with Path(str(prefix) + suffix).open('x') as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
    sources = list((ROOT / 'src/qwen_booster').glob('*.py')) + [Path(__file__), ROOT / 'scripts/evaluate_api.py', ROOT / 'scripts/audit_workload_tokens.py', SUITE, Path(str(original) + '-responses.jsonl')]
    save('-config.json', {'command': command, 'environment_overrides': config['environment_overrides'], 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'snapshots_before': {str(p): p.stat().st_size for p in files}, 'scope': __doc__})
    observations = []
    for name, case in zip(('a_initial', 'b_initial'), cases):
        token_audit = count_case(tokenizer, case, 262144, 204800)
        assert token_audit['fits_native_context'] and token_audit['meets_minimum_input']
        start = time.monotonic()
        with Path(str(prefix) + '-' + name + '-server.log').open('x') as log:
            server = subprocess.Popen(command, env={**os.environ, **config['environment_overrides']}, stdout=log, stderr=subprocess.STDOUT)
            monitor = None
            try:
                monitor = subprocess.Popen(['.venv/bin/python', 'scripts/monitor_memory.py', '--pid', str(server.pid), '--output', str(prefix) + '-' + name + '-memory.jsonl'])
                print('loading', name, server.pid, flush=True)
                endpoint = 'http://127.0.0.1:18083'
                def get(path):
                    with urllib.request.urlopen(endpoint + path, timeout=10) as response:
                        return json.load(response)
                while True:
                    if server.poll() is not None or time.monotonic() - start > 1200:
                        raise RuntimeError('Server failed to load')
                    try:
                        if get('/health').get('loaded_model') == run['model']:
                            break
                    except (OSError, TimeoutError):
                        pass
                    time.sleep(2)
                startup = time.monotonic() - start
                before = get('/v1/cache/stats')
                assert before['exact_resident_bytes'] == 0 and before['disk_hits'] == 0
                print('restoring', name, flush=True)
                result = evaluate(endpoint + '/v1', case, 7200, model=run['model'], include_usage=True)
                after = get('/v1/cache/stats')
                reference = cold[name]['result']
                checks = {
                    'complete': result['complete'] and result['finish_reason'] == 'stop' and not result['server_errors'],
                    'correct': grade_answer(result['text'], case['grading']) is True,
                    'bare_json': grade_json_format(result['text'], case['grading']) is True,
                    'parity': all(result[k] == reference[k] for k in ('text', 'reasoning')) and result['usage']['completion_tokens'] == reference['usage']['completion_tokens'],
                    'full_input': result['usage']['prompt_tokens'] == reference['usage']['prompt_tokens'] == token_audit['actual_prompt_tokens'],
                    'full_cache': result['server_timings']['cache_n'] == token_audit['actual_prompt_tokens'] - 1,
                    'disk_hit': after['disk_hits'] > before['disk_hits'],
                    'cold_reference': reference['server_timings']['cache_n'] == 0,
                }
                row = {'name': name, 'pid': server.pid, 'startup_seconds': startup, 'input_audit': token_audit, 'result': result, 'before': before, 'after': after, 'checks': checks, 'cold_reference_seconds': reference['seconds']}
                observations.append(row)
                save('-' + name + '.json', row)
                print('finished', name, result['seconds'], checks, flush=True)
            finally:
                if server.poll() is None:
                    server.terminate()
                server.wait(timeout=120)
                if monitor:
                    monitor.wait(timeout=30)
    save('-review.json', {'bounded_disk_restart_passed': all(all(r['checks'].values()) for r in observations) and len({r['pid'] for r in observations}) == 2, 'observations': observations, 'acceptance_complete': False, 'scope': 'Two synthetic 200K fixtures, independent fresh processes, prior identical cold references; excludes startup from request latency. No broader quality or stable speed claim.'})


if __name__ == '__main__':
    main()
