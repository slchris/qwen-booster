import importlib.util
import io
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "evaluate_api", Path(__file__).parents[1] / "scripts/evaluate_api.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def response(*chunks, done=True):
    events = [b"data: " + json.dumps({"choices": [chunk]}).encode() + b"\n\n"
              for chunk in chunks]
    if done:
        events.append(b"data: [DONE]\n\n")
    return io.BytesIO(b"".join(events))


def test_role_is_not_first_content_and_retrieval_is_not_quality(monkeypatch):
    monkeypatch.setattr(evaluator.urllib.request, "urlopen", lambda *a, **k: response(
        {"delta": {"role": "assistant"}},
        {"delta": {"content": "secret"}},
        {"delta": {}, "finish_reason": "stop"}))
    result = evaluator.evaluate("http://localhost/v1",
        {"messages": [], "expected_substrings": ["secret"]}, 1)
    assert result["complete"] and result["retrieval_check_passed"]
    assert len(result["content_chunk_arrivals"]) == 1
    assert result["quality_review_required"]


def test_truncated_stream_and_inline_thinking(monkeypatch):
    monkeypatch.setattr(evaluator.urllib.request, "urlopen", lambda *a, **k: response(
        {"delta": {"content": "<think>work"}}, done=False))
    result = evaluator.evaluate("http://localhost/v1", {"messages": []}, 1)
    assert not result["complete"]
    assert result["first_visible_answer_seconds"] is None
    assert result["retrieval_check_passed"] is None


def test_usage_only_tail_preserves_completion_and_speculation_metrics(monkeypatch):
    def open_response(request, **kwargs):
        assert json.loads(request.data)["stream_options"] == {"include_usage": True}
        events = [
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 2},
             "timings": {"draft_rounds": 1, "draft_n": 2, "draft_n_accepted": 1}},
        ]
        return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n"
                                  for e in events) + b"data: [DONE]\n\n")
    monkeypatch.setattr(evaluator.urllib.request, "urlopen", open_response)
    result = evaluator.evaluate("http://localhost/v1", {"messages": []}, 1,
                                include_usage=True)
    assert result["complete"] and result["text"] == "ok"
    assert result["usage"]["completion_tokens"] == 2
    assert result["server_timings"]["draft_n_accepted"] == 1


def test_sse_server_error_is_preserved_and_never_complete(monkeypatch):
    payload = (b'data: {"error": {"message": "Timed out waiting for 600s"}}\n\n'
               b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
               b'data: [DONE]\n\n')
    monkeypatch.setattr(evaluator.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    result = evaluator.evaluate('http://localhost/v1', {'messages': []}, 1)
    assert not result['complete']
    assert result['server_errors'] == [{'message': 'Timed out waiting for 600s'}]


def test_streamed_json_content_and_format_are_reported_separately(monkeypatch):
    monkeypatch.setattr(evaluator.urllib.request, "urlopen", lambda *a, **k: response(
        {"delta": {"content": '```json\n{"answer":'}},
        {"delta": {"content": '3}\n```'}},
        {"delta": {}, "finish_reason": "stop"}))
    result = evaluator.evaluate("http://localhost/v1", {
        "messages": [], "grading": {"kind": "json", "expected": {"answer": 3}}}, 1)
    assert result["complete"]
    assert result["task_check_passed"] is True
    assert result["bare_json_format_passed"] is False
    assert result["text"] == '```json\n{"answer":3}\n```'
