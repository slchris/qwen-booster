import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'benchmark_concurrency', Path(__file__).parents[1] / 'scripts/benchmark_concurrency.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(level, ttft, tps, tokens=64, digest='a'):
    return {'level': level, 'ttft_seconds': ttft, 'decode_tps': tps,
            'tokens': tokens, 'text_sha256': digest}


def test_level_summary_medians_and_text_identity():
    rows = [row(1, 2.0, 10.0), row(1, 3.0, 12.0),
            row(2, 4.0, 6.0), row(2, 5.0, 8.0, digest='b')]
    summary = module.level_summary(rows)
    assert summary['1']['ttft_median_seconds'] == 2.5
    assert summary['1']['decode_tps_median'] == 11.0
    assert summary['1']['tokens_total'] == 128
    assert summary['1']['texts_identical'] is True
    assert summary['2']['texts_identical'] is False


def test_level_summary_skips_errors_and_reports_empty_levels():
    rows = [row(1, 2.0, 10.0), {'level': 2, 'error': 'timeout'}]
    summary = module.level_summary(rows)
    assert summary['2'] == {'completed': 0}
