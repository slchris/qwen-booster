import sys

import pytest

from qwen_booster import vlm_server


@pytest.mark.parametrize('timeout', ['0', '-1', 'nan', 'inf'])
def test_invalid_token_deadline_rejected_before_model_loading(monkeypatch, timeout):
    monkeypatch.setattr(sys, 'argv', ['vlm_server', '--model', 'unused', '--token-timeout', timeout])
    with pytest.raises(SystemExit) as error:
        vlm_server.main()
    assert error.value.code == 2


def test_27b_borrow_flag_rejects_flash_before_backend_import(monkeypatch):
    from types import SimpleNamespace
    from qwen_booster import models, context
    monkeypatch.setattr(models, 'resolve_local_model', lambda _: ('unused', {}, SimpleNamespace(id='qwen3.8-flash-next')))
    monkeypatch.setattr(context.ContextBudget, 'from_config', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['vlm_server', '--model', 'unused', '--borrow-27b-checkpoint-row'])
    with pytest.raises(SystemExit) as error:
        vlm_server.main()
    assert error.value.code == 2


def test_server_argv_passes_host_concurrency_and_api_key():
    from types import SimpleNamespace
    args = SimpleNamespace(host='0.0.0.0', port=18081, max_num_seqs=4, context=16384,
                           api_key='secret', draft=None, block=3)
    argv = vlm_server.server_argv('/tmp/model', args)
    assert argv[argv.index('--host') + 1] == '0.0.0.0'
    assert argv[argv.index('--max-num-seqs') + 1] == '4'
    assert argv[argv.index('--api-key') + 1] == 'secret'


def test_server_argv_defaults_stay_local_and_single_sequence():
    from types import SimpleNamespace
    args = SimpleNamespace(host='127.0.0.1', port=18081, max_num_seqs=1, context=32768,
                           api_key=None, draft=None, block=3)
    argv = vlm_server.server_argv('/tmp/model', args)
    assert '--api-key' not in argv
    assert argv[argv.index('--max-num-seqs') + 1] == '1'
