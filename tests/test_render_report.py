import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'render_report', Path(__file__).parents[1] / 'scripts/render_report.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_render_bars_metadata_and_escaping():
    metadata = {'type': 'metadata', 'model_path': '<b>x</b>', 'note': 1, 'nested': {'a': 1}}
    rows = [{'type': 'measurement', 'case': 'cold', 'prepare_seconds': 10.0, 'ok': True},
            {'type': 'measurement', 'case': 'warm', 'prepare_seconds': 0.5, 'ok': False}]
    text = module.render(metadata, rows, 'demo <script>')
    assert 'demo &lt;script&gt;' in text
    assert '&lt;b&gt;x&lt;/b&gt;' in text
    assert 'nested' not in text
    assert text.count('width:100.0%') == 1
    assert 'width:5.0%' in text
    assert '>true<' in text and '>false<' in text
    assert module.columns(rows)[:2] == ['case', 'type']


def test_load_rows_splits_metadata(tmp_path):
    path = tmp_path / 'run.jsonl'
    path.write_text('{"type": "metadata", "label": "x"}\n'
                    '{"type": "measurement", "case": "cold"}\n')
    metadata, rows = module.load_rows(path)
    assert metadata == {'type': 'metadata', 'label': 'x'}
    assert rows == [{'type': 'measurement', 'case': 'cold'}]
