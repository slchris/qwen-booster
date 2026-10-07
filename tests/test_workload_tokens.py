import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('audit_workload_tokens',
    Path(__file__).parents[1] / 'scripts' / 'audit_workload_tokens.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_actual_template_count_controls_context_gate_without_changing_messages():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'identical task'}]
            assert kwargs['enable_thinking'] is True
            assert kwargs['preserve_thinking'] is False
            return list(range(201))

    case = {'id': 'case', 'messages': [{'role': 'user', 'content': 'identical task'}],
            'prompt_tokens': 190, 'max_tokens': 10, 'enable_thinking': True,
            'preserve_thinking': False}
    row = audit.count_case(Tokenizer(), case, native_context=210, minimum_input=200)
    assert row['actual_prompt_tokens'] == 201
    assert row['source_prompt_tokens'] == 190
    assert row['meets_minimum_input']
    assert not row['fits_native_context']
    assert row['remaining_history_tokens'] == -1
    assert case['prompt_tokens'] == 190
