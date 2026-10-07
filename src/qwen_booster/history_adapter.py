"""Preserve reasoning metadata through the pinned MLX-VLM message formatter."""

import functools
import sys


def wrap_template(original, render):
    @functools.wraps(original)
    def apply(processor, config, prompt, add_generation_prompt=True,
              return_messages=False, num_images=0, num_audios=0, **kwargs):
        model_type = config.get('model_type') if isinstance(config, dict) else getattr(config, 'model_type', None)
        source = [prompt] if isinstance(prompt, dict) else prompt
        preserve = (model_type in ('qwen3_5', 'qwen4_exp') and isinstance(source, list)
                    and any(isinstance(m, dict) and
                            ('reasoning_content' in m or 'reasoning' in m) for m in source))
        options = dict(add_generation_prompt=add_generation_prompt,
                       num_images=num_images, num_audios=num_audios, **kwargs)
        if not preserve:
            return original(processor, config, prompt, return_messages=return_messages, **options)
        messages = original(processor, config, prompt, return_messages=True, **options)
        if len(messages) != len(source):
            raise ValueError('Cannot preserve history: formatter changed message count')
        messages = [dict(m) for m in messages]
        for before, after in zip(source, messages):
            if not isinstance(before, dict):
                continue
            if before.get('role', 'user') != after.get('role'):
                raise ValueError('Cannot preserve history: formatter changed message roles')
            for field in ('reasoning_content', 'reasoning'):
                if field in before:
                    after[field] = before[field]
        if return_messages:
            return messages
        return render(processor, messages, add_generation_prompt, **kwargs)
    return apply


def install():
    from mlx_vlm import prompt_utils
    from mlx_vlm.server import request_normalization
    from mlx_vlm.server.generation import GenerationArguments

    if getattr(GenerationArguments, '_qwen_history_adapter', False):
        return
    original = prompt_utils.apply_chat_template
    wrapped = wrap_template(original, prompt_utils.get_chat_template)
    # Routes and server compatibility exports capture this function on import.
    for name, module in list(sys.modules.items()):
        if name == 'mlx_vlm.prompt_utils' or name.startswith('mlx_vlm.server'):
            if getattr(module, 'apply_chat_template', None) is original:
                module.apply_chat_template = wrapped

    build = request_normalization._build_gen_args
    template = GenerationArguments.to_template_kwargs

    @functools.wraps(build)
    def build_with_history(request, *args, **kwargs):
        option = getattr(request, 'preserve_thinking', None)
        if option is not None and type(option) is not bool:
            raise ValueError('preserve_thinking must be a boolean')
        result = build(request, *args, **kwargs)
        result._qwen_preserve_thinking = option
        return result

    @functools.wraps(template)
    def template_with_history(self):
        result = template(self)
        option = getattr(self, '_qwen_preserve_thinking', None)
        if option is not None:
            result['preserve_thinking'] = option
        return result

    request_normalization._build_gen_args = build_with_history
    GenerationArguments.to_template_kwargs = template_with_history
    GenerationArguments._qwen_history_adapter = True
