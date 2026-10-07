from qwen_booster.metrics import recommend, summarize


def row(length, step, ttft):
    return dict(prompt_tokens=length, prefill_step_size=step,
                ttft_seconds=ttft, prefill_tps=length / ttft,
                decode_tps=20.0, peak_memory_gb=16.0)


def test_medians_are_grouped_by_length_and_configuration():
    values = summarize([row(512, 2048, 1), row(512, 2048, 9),
                        row(512, 2048, 2), row(2048, 2048, 8)])
    assert values[0]["ttft_seconds"] == 2
    assert values[0]["runs"] == 3
    assert values[1]["ttft_seconds"] == 8


def test_recommendation_never_compares_different_prompt_lengths():
    values = summarize([row(2048, 2048, 2), row(2048, 512, 1), row(8192, 512, 9)])
    result = recommend(values)
    assert len(result) == 1
    assert result[0]["recommended_step"] == 512
    assert result[0]["ttft_speedup"] == 2


def test_no_baseline_means_no_speedup_claim():
    assert recommend(summarize([row(512, 512, 1)])) == []


def test_equivalent_prefill_sizes_do_not_produce_false_speedups():
    result = recommend(summarize([row(512, 512, 1), row(512, 2048, 2)]))
    assert result[0]["recommended_step"] == 2048
    assert result[0]["ttft_speedup"] == 1
