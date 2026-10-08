# Third-party notices

This project is licensed under the MIT License (see `LICENSE`). It builds on,
links against, or documents third-party software and model weights that carry
their own licenses. This file lists them for convenience; the authoritative
license text ships with each dependency.

## Runtime dependencies

| Component | License | Copyright / source |
| --- | --- | --- |
| MLX | MIT | Copyright © 2023 Apple Inc., https://github.com/ml-explore/mlx |
| MLX-LM | MIT | Copyright © 2023 Apple Inc., https://github.com/ml-explore/mlx-lm |
| MLX-VLM | MIT | Copyright © 2025 Prince Canuma, https://github.com/Blaizzy/mlx-vlm |
| Transformers, Tokenizers, Safetensors | Apache-2.0 | Hugging Face, https://github.com/huggingface |

The serving path in this repository is an experiment layer on top of a pinned
MLX-VLM revision (`requirements-vlm.lock`). It does not vendor MLX-VLM code.

## Model weights (not distributed here)

| Model | License | Source |
| --- | --- | --- |
| Qwen/Qwen3.8-27B | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.8-27B |
| mlx-community/Qwen3.8-27B-4bit | Apache-2.0 (derivative) | https://huggingface.co/mlx-community/Qwen3.8-27B-4bit |
| Qwen/Qwen3.8-Flash-Next | Qwen Community License 1.0 | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| mlx-community/Qwen3.8-Flash-Next-4bit | Qwen Community License 1.0 (derivative) | https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit |
| z-lab/Qwen3.8-27B-DFlash2 | Apache-2.0 | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 |

No model weights, MTP drafts, or DFlash checkpoints are distributed with this
repository. Users download them separately and are bound by the license of
each model.

## Benchmark evidence

The `benchmarks/` directory archives raw experiment outputs. Model responses
embedded there are model outputs, not project content, and retain the
applicable model license. Archived server logs may contain absolute file
paths and timestamps from the machine that produced them.
