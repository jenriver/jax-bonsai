# Gemma3 in JAX

This directory contains a pure JAX implementation of the [Gemma3 language model](https://deepmind.google/models/gemma/gemma-3/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.



## Model Family and Hardware Compatibility Matrix  
*(Last Updated: 2025-06-25)*

From the full released [Gemma3 checkpoints](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d).

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Models (Instruction-Tuned)** | | | | | | | | |
| [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | ✅ Supported | ✅ Runs | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)  | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | ✅ Supported | ❔ Needs check | ❔ Needs check| ❔ Needs check| ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| **Dense Models (Pre-Trained)** | | | | | | | | |
| [gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt)  | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [gemma-3-27b-pt](https://huggingface.co/google/gemma-3-27b-pt) | ✅ Supported | ❔ Needs check | ❔ Needs check| ❔ Needs check| ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [model.py](model.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [test_model.py](tests/test_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.