
<h1 align="center">🌳JAX Bonsai, a curation of SoTA models in JAX🌳</h1>


A collection of exemplary JAX models using JAX's [NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html). This repository serves as a curated list of well-documented and easy-to-understand JAX implementations of common machine learning models.

We support integration with powerful post-training libraries such as [Tunix](https://github.com/google/tunix/tree/main).

## Current Models

* [Qwen 3](https://github.com/jenriver/jax-bonsai/tree/main/bonsai/models/qwen3)
* (Coming soon) Llama 3
* (Coming soon) Gemma 3
* (Coming soon) SAM2

## 🏁 Getting Started

To get started with JAX Bonsai, follow these steps to set up your development environment and run the models.

### 1. Clone the Repository

First, clone the JAX Bonsai repository to your local machine:

```bash
git clone https://github.com/jenriver/jax-bonsai.git
cd jax-bonsai
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies. You can use either `venv` (built-in with Python) or `uv` (a fast Python package installer and dependency manager).

```bash
# venv
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# uv (for faster dependency resolution and installation)
uv venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

Once your virtual environment is activated, install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run a Model Example

Navigate to a specific model's directory (e.g., `bonsai/models/qwen3`) and follow the instructions in its [`README.md`](bonsai/models/qwen3/README.md) to run an example or notebook.

```bash
cd bonsai/models/qwen3
# Follow instructions in qwen3/README.md
```

## Contributing

We welcome contributions to JAX Bonsai! If you're interested in adding new models, improving existing implementations, or enhancing documentation, please refer to our [Contributing Guidelines](CONTRIBUTING.md).