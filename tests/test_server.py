import pytest

from qwen_booster.server import validate_request


def test_text_chat_request():
    messages = [{"role": "user", "content": "你好"}]
    assert validate_request({"messages": messages, "stream": True}) == (messages, 256, True, False)


@pytest.mark.parametrize("extra", [{"temperature": -1}, {"tools": []}, {"max_tokens": True},
                                   {"max_tokens": -1}, {"stream": "true"}, {"model": "other"}])
def test_unsupported_settings_are_not_silently_ignored(extra):
    with pytest.raises(ValueError):
        validate_request({"messages": [{"role": "user", "content": "hi"}], **extra})


@pytest.mark.parametrize("messages", [[], "hello", [None], [{"role": "tool", "content": "x"}],
                                       [{"role": "user", "content": [{"type": "image"}]}]])
def test_reject_non_text_messages(messages):
    with pytest.raises(ValueError):
        validate_request({"messages": messages})
