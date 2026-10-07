"""Validated native sampling, including full-history penalties after cache restore."""

import math


FIELDS = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
          "repetition_penalty", "seed"}


def settings(body):
    result = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
              "presence_penalty": 0.0, "repetition_penalty": 1.0}
    result.update({key: body[key] for key in FIELDS if key in body})
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1), ("min_p", 0, 1),
                           ("presence_penalty", -2, 2), ("repetition_penalty", 0.01, 2)):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
    if result["top_p"] == 0:
        raise ValueError("top_p must be greater than zero")
    if type(result["top_k"]) is not int or not 0 <= result["top_k"] <= 248320:
        raise ValueError("top_k must be an integer in [0, 248320]")
    if "seed" in result and (type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32):
        raise ValueError("seed must be an unsigned 32-bit integer")
    return result


def native_options(config, prompt_tokens, max_tokens, prefilled=True):
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    if "seed" in config:
        mx.random.seed(config["seed"])
    sampler = make_sampler(temp=config["temperature"], top_p=config["top_p"],
                           min_p=config["min_p"], top_k=config["top_k"])
    processors = make_logits_processors(
        presence_penalty=config["presence_penalty"],
        presence_context_size=len(prompt_tokens) + max_tokens,
        repetition_penalty=config["repetition_penalty"] if config["repetition_penalty"] != 1 else None,
        repetition_context_size=len(prompt_tokens) + max_tokens)
    if not processors:
        return {"sampler": sampler, "logits_processors": []}
    # generate_step sees only the final input token after Engine.prepare().
    # Restore the full logical history for penalties independently of cache hits.
    prefix = mx.array(prompt_tokens[:-1], dtype=mx.uint32)
    def with_history(processor):
        def apply(tokens, logits):
            return processor(mx.concatenate([prefix, tokens]), logits)
        return apply
    return {"sampler": sampler,
            "logits_processors": [with_history(p) for p in processors] if prefilled else processors}
