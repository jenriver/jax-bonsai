# Qwen3 in JAX NNX

This directory contains a pure JAX implementation of the Qwen3 language model, using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.


## Qwen3 Support status
*(Last Updated: 2025-06-11)*

### Key Features

* **Pure JAX/NNX:** A clean, modern implementation for clarity and performance.
* **Built-in Model Configurations:** Includes pre-defined `ModelConfig` classes in [model.py](model.py) for various Qwen3 model sizes.
* **Sharding-Aware:** Features a `ShardingConfig` dataclass to easily manage Tensor Parallelism (TP) and Fully Sharded Data Parallelism (FSDP) across model weights and activations.
* **Core Transformer Components:** Provides clear, modular implementations of essential transformer blocks:
    * `Attention`: Grouped-Query Attention (GQA) with RoPE embeddings.
    * `MLP`: Gated feed-forward network with SiLU activation.
    * `RMSNorm`: Standard Root Mean Square Normalization.
    * `Embedder`: Handles token embedding and final decoding/logits projection.

 
### Model Family Support Matrix

| Model Name | Architecture | Total Parameters | Active Parameters | Supported |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3-0.6B** | Dense | 0.6 Billion | ~0.6 Billion | ✅ [Yes](https://github.com/jenriver/bonsai/tree/main/bonsai/models/qwen3) |
| **Qwen3-1.7B** | Dense | 1.7 Billion | ~1.7 Billion | ✅ [Yes]() |
| **Qwen3-4B** | Dense | 4 Billion | ~4 Billion | ✅ [Yes]() |
| **Qwen3-8B** | Dense | 8 Billion | ~8 Billion | ✅ [Yes]() |
| **Qwen3-14B** | Dense | 14 Billion | ~14 Billion | ❌ No |
| **Qwen3-32B** | Dense | 32 Billion | ~32 Billion | ❌ No |
| **Qwen3-30B-A3B** | MoE | 30 Billion | ~3 Billion | ❌ No |
| **Qwen3-235B-A22B** | MoE | 235 Billion | ~22 Billion | ❌ No |


### Hardware Compatibility Matrix


| Hardware | Status | Notes |
| :--- | :--- | :--- |
| **CPU** | ✅ Working | Initial setup complete. Performance benchmarks pending. |
| **GPU A100 (1x)** | ✅ Working | Initial setup complete. Performance benchmarks pending. |
| **GPU H100 (1x)** | ✅ Working | Initial setup complete. Performance benchmarks pending. |
| **GPU A100 (8x)** | 🟡 **In Progress** | Code runnable, Multi-chip sharding not yet supported. |
| **GPU H100 (8x)** | 🟡 **In Progress** | Code runnable, Multi-chip sharding not yet supported. |
| **TPU v2 (8x)** | ❌ **Needs Work**| Multi-chip sharding not yet supported. |
| **TPU v5e (1x)** | ❌ **Needs Work** | Multi-chip sharding not yet supported. |

## Contribute to this model!

We welcome contributions! You can contribute to this model via the following:
* **Model family coverage support** Support Qwen3's [Model Family variants](#model-family-support-matrix) by adding appropriate [model.py](model.py), test examples, and updating this matrix. [Example PR]()
* **Hardware coverage support** Support this model to run on different hardwares and update this matrix. [Example PR]()