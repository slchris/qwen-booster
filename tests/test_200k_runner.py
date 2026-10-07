import importlib.util
import io
import json
import sys
from pathlib import Path
import pytest


def test_quality_failure_is_retained_and_final_replay_uses_identical_history(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('large_runner', scripts / 'evaluate_200k.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = {'id': 'large', 'fact_position_fraction': 0.95, 'prompt_tokens': 204801,
               'headroom_tokens': 4096, 'expected_substrings': ['secret'],
               'messages': [{'role': 'user', 'content': '\n'.join(
                   f'记录 {i:06d}: 项目编号 {10000000+i};' for i in range(100))}]}
    suite = tmp_path / 'suite.jsonl'
    suite.write_text(json.dumps(fixture))
    calls = []

    def evaluate(endpoint, case, timeout, **kwargs):
        calls.append(case)
        answer = 'wrong' if len(calls) == 2 else case['grading']['expected']
        return {'complete': True, 'finish_reason': 'stop', 'text': answer, 'reasoning': '',
                'task_check_passed': answer == case['grading']['expected'],
                'first_visible_answer_seconds': 1,
                'server_timings': {'cache_n': 204801 if len(calls) in (2, 3, 4) else 0}}

    monkeypatch.setattr(module, 'evaluate', evaluate)
    monkeypatch.setattr(module, 'reset', lambda endpoint: None)
    output = tmp_path / 'out.jsonl'
    monkeypatch.setattr(sys, 'argv', ['evaluate_200k', '--suite', str(suite), '--output', str(output),
                                    '--endpoint', 'test', '--model', 'model', '--backend', 'vlm', '--turns', '4'])
    module.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(calls) == 5
    assert calls[-1] == calls[-2]
    assert calls[2]['messages'][-2]['content'] == 'wrong'
    assert rows[2]['result']['task_check_passed'] is False
    assert all(row['reuse_passed'] for row in rows if row['type'] == 'measurement')
    assert rows[-1]['parity_passed'] is True

    # A lost snapshot must stop the optional diagnostic run before another
    # expensive cold prefill is silently issued.
    calls.clear()
    monkeypatch.setattr(module.urllib.request, 'urlopen',
                        lambda *a, **k: io.BytesIO(b'{"exact_resident_bytes": 0}'))
    monkeypatch.setattr(sys, 'argv', ['evaluate_200k', '--suite', str(suite),
                                    '--output', str(tmp_path / 'lost.jsonl'),
                                    '--endpoint', 'test', '--model', 'model', '--backend', 'vlm',
                                    '--turns', '4', '--stop-on-cache-loss'])
    with pytest.raises(RuntimeError, match='retained no exact APC snapshot'):
        module.main()
    assert len(calls) == 1
