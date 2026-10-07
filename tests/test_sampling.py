import pytest

from qwen_booster.sampling import settings
from qwen_booster.server import validate_request


@pytest.mark.parametrize("field,value", [("temperature", float("nan")), ("temperature", True),
    ("top_p", 0), ("top_k", 1.5), ("min_p", 2), ("seed", -1),
    ("repetition_penalty", 0), ("presence_penalty", float("inf"))])
def test_invalid_sampling_parameters(field, value):
    with pytest.raises(ValueError):
        settings({field: value})


def test_quality_sampling_and_preserved_reasoning_are_accepted():
    request = {"messages": [{"role": "assistant", "content": "answer",
                             "reasoning_content": "prior reasoning"}],
               "temperature": 1.0, "top_p": 0.95, "top_k": 20,
               "enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"}
    assert validate_request(request)[3] is True
    assert settings(request)["top_k"] == 20
