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
