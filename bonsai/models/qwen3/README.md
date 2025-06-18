# Qwen3 in JAX

This directory contains a pure JAX implementation of the [Qwen3 language model](https://qwenlm.github.io/blog/qwen3/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.



## Model Family and Hardware Compatibility Matrix  
*(Last Updated: 2025-06-17)*

 

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Models** | | | | | | | | |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | ✅ Supported | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | ✅ Supported | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 🟡 Not started| ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 🟡 Not started | ❔ Needs check | ❔ Needs check| ❔ Needs check| ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| **MoE Models** | | | | | | | | |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [model.py](model.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [test_model.py](tests/test_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.