# Bonsai🌳

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A collection of State-of-the-Art models implemented using JAX [NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html). This repository serves as a curated list of well-documented and easy-to-understand JAX implementations of common machine learning models.


Bonsai supports integration with powerful JAX libraries.
* [Tunix](https://github.com/google/tunix/tree/main), a post-training library supporting Supervised Fine-Tuning, RL, Knoweldge Distillation.

## Current Models

* [Qwen 3](bonsai/models/qwen3/README.md)
* [Llama 3](bonsai/models/llama3/README.md)
* [Gemma 3](bonsai/models/gemma3/README.md)
* (Coming soon) SAM2

## 🏁 Getting Started

To get started with JAX Bonsai, follow these steps to set up your development environment and run the models.

### Installation

Clone the JAX Bonsai repository to your local machine.

```bash
git clone https://github.com/jenriver/bonsai.git
cd bonsai
```

Install the latest repository.
```bash
pip install -e .
```

### Running models

Bonsai models can be imported and used like the following:

```
from flax import nnx
from bonsai.models.qwen3 import model, params

MODEL_CP_PATH = $YOUR_CKPT

config = model.ModelConfig.qwen3_0_6_b()
qwen3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)
```

Jump right into our [Colab notebook](https://colab.sandbox.google.com/github/jenriver/bonsai/blob/qwen3/bonsai/models/qwen3/qwen3_example.ipynb) to see Qwen3 in action.

You can also run the following to benchmark and see generate results from a Qwen3 model.
```python
python -m bonsai.models.qwen3.tests.test_qwen3
```


## Contributing

We welcome contributions!
If you're interested in adding new models, improving existing implementations, or enhancing documentation, please see our [Contributing Guidelines](CONTRIBUTING.md).

## Useful Links
* [JAX](https://docs.jax.dev/en/latest/quickstart.html): Learn more about JAX, a super fast NumPy-based ML framework with automatic differentiation.
* [The JAX ecosystem](https://docs.jaxstack.ai/en/latest/getting_started.html): Unlock unparalleled speed and scale for your next-generation models. Explore an incredible suite of tools and libraries that effortlessly extend JAX's capabilities, transforming how you build, train, and deploy.