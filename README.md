
<h1 align="center">🌳JAX Bonsai, a curation of SoTA models in JAX🌳</h1>


A collection of exemplary JAX models using JAX's [NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html). This repository serves as a curated list of well-documented and easy-to-understand JAX implementations of common machine learning models.

We support integration with powerful post-training libraries such as [Tunix](https://github.com/google/tunix/tree/main).

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

Try out our [colab](https://colab.sandbox.google.com/github/jenriver/bonsai/blob/qwen3/bonsai/models/qwen3/qwen3_example.ipynb) on running a Qwen3 generate.

You can also run the following to benchmark and see generate results from a Qwen3 model.
```python
python -m bonsai.models.qwen3.tests.test_qwen3
```


## Contributing

We welcome contributions!
If you're interested in adding new models, improving existing implementations, or enhancing documentation, please see our [Contributing Guidelines](CONTRIBUTING.md).