"""Context admission independent of GPU allocation or quality claims."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    native_limit: int
    service_limit: int

    @classmethod
    def from_config(cls, config, requested):
        text = config.get("text_config", config)
        native = text.get("max_position_embeddings")
        if type(native) is not int or native <= 0:
            raise ValueError("model must declare a positive max_position_embeddings")
        if type(requested) is not int or not 0 < requested <= native:
            raise ValueError(f"context must be in [1, {native}]; extended context is not validated")
        return cls(native, requested)

    def admit(self, prompt_tokens, output_tokens):
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt must contain tokens")
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError("output reservation must be positive")
        if prompt_tokens + output_tokens > self.service_limit:
            raise ValueError(
                f"prompt ({prompt_tokens}) plus reserved output ({output_tokens}) "
                f"exceeds context budget {self.service_limit}; input was not truncated"
            )


def full_attention_kv_bytes(config, tokens, bytes_per_element=2):
    """Logical full-attention KV only; excludes state, padding, copies and workspace."""
    text = config.get("text_config", config)
    layers = text.get("layer_types")
    if layers is None:
        raise ValueError("explicit layer_types required for hybrid KV estimate")
    count = sum(layer == "full_attention" for layer in layers)
    return (count * 2 * text["num_key_value_heads"] * text["head_dim"]
            * tokens * bytes_per_element)
