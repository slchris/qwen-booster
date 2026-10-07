import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'benchmark_decode_contexts', Path(__file__).parents[1] / 'scripts/benchmark_decode_contexts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_tile_tokens_is_exact_and_deterministic():
    unit = [10, 11, 12]
    assert module.tile_tokens(unit, 7) == [10, 11, 12, 10, 11, 12, 10]
    assert module.tile_tokens(unit, 3) == unit
    try:
        module.tile_tokens([], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("empty unit must fail")


def test_build_prompt_roundtrips_through_a_real_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    tokenizer = Tokenizer(models.WordLevel({"hello": 0, "world": 1}, unk_token="hello"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    assert module.build_prompt(tmp_path, 5, corpus="hello world") == "hello world hello world hello"
    assert module.build_prompt(tmp_path, 4, corpus="hello world") == "hello world hello world"


def test_summarize_groups_prime_and_decode_rows():
    rows = [
        {"phase": "prime", "target": 32768, "seconds": 90.0,
         "usage": {"prompt_tokens": 32770}},
        {"phase": "decode", "target": 32768, "seconds": 9.0, "ttft_seconds": 0.5,
         "decode_tps": 15.0, "text_sha256": "a",
         "usage": {"prompt_tokens_details": {"cached_tokens": 32769}}},
        {"phase": "decode", "target": 32768, "seconds": 9.5, "ttft_seconds": 0.6,
         "decode_tps": 17.0, "text_sha256": "a",
         "usage": {"prompt_tokens_details": {"cached_tokens": 32769}}},
    ]
    summary = module.summarize(rows)["32768"]
    assert summary["prime_seconds"] == 90.0
    assert summary["prompt_tokens"] == 32770
    assert summary["cached_tokens"] == 32769
    assert summary["decode_tps_median"] == 16.0
    assert summary["texts_identical"] is True
