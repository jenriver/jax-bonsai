# Llama3 in JAX

This directory contains a pure JAX implementation of the [Llama3 language model](https://www.llama.com/models/llama-3/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.



## Model Family and Hardware Compatibility Matrix  
*(Last Updated: 2025-06-17)*

From the full released [Llama3 checkpoints](https://huggingface.co/models?other=llama-3&sort=downloads).

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Models (Instruction-Tuned)** | | | | | | | | |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | ✅ Supported| ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| **Dense Models (Pre-Trained)** | | | | | | | | |
| [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [model.py](model.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [test_model.py](tests/test_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.