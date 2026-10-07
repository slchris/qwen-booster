import pytest

from qwen_booster.reasoning import ReasoningParser


@pytest.mark.parametrize("width", range(1, 13))
def test_delimiter_split_across_arbitrary_chunks(width):
    source = "想清楚再回答</think>\n\n答案是42。"
    parser = ReasoningParser(True)
    parts = [parser.feed(source[i:i+width]) for i in range(0, len(source), width)]
    parts.append(parser.feed("", final=True))
    assert "".join(p.get("reasoning_content", "") for p in parts) == "想清楚再回答"
    assert "".join(p.get("content", "") for p in parts) == "\n\n答案是42。"


def test_truncated_reasoning_is_not_presented_as_an_answer():
    parser = ReasoningParser(True)
    assert parser.feed("还在推理</thi") == {"reasoning_content": "还在推理"}
    assert parser.feed("", final=True) == {"reasoning_content": "</thi"}


def test_non_thinking_content_is_preserved():
    assert ReasoningParser(False).feed("literal </think>", final=True) == {"content": "literal </think>"}
