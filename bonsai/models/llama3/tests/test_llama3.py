# This file should be run as a module from the project root using:
# python -m bonsai.models.llama3.tests.test_llama3

import os
import sys
import time

from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.generate import sampler
from bonsai.models.llama3 import model
from bonsai.models.llama3 import params


# model_name = "meta-llama/Llama-3.2-1B"
# MODEL_CP_PATH = "/tmp/llama3.2-1b-weights" # Specify your desired download directory
"""
jax._src.interpreters.xla.InvalidInputException: Argument 'ShapeDtypeStruct(shape=(2048, 8, 64), dtype=float32)' of type <class 'jax._src.api.ShapeDtypeStruct'> is not a valid JAX type.
"""
# model_name = "meta-llama/Meta-Llama-3-8B"
# MODEL_CP_PATH = "/tmp/llama3-8b-weights" # Specify your desired download directory
"""
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
"""
# model_name = "meta-llama/Llama-3.1-8B"
# MODEL_CP_PATH = "/tmp/llama3.1-8b-weights" # Specify your desired download directory
"""
jax._src.interpreters.xla.InvalidInputException: Argument 'ShapeDtypeStruct(shape=(4096, 8, 128), dtype=float32)' of type <class 'jax._src.api.ShapeDtypeStruct'> is not a valid JAX type.
"""
model_name = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_CP_PATH = "/tmp/llama3.2-3b-instruct"
"""
jax._src.interpreters.xla.InvalidInputException: Argument 'ShapeDtypeStruct(shape=(3072, 128256), dtype=float32)' of type
 <class 'jax._src.api.ShapeDtypeStruct'> is not a valid JAX type.
"""

if os.path.isdir(MODEL_CP_PATH):
    print(f"'{MODEL_CP_PATH}' exists, skipping huggingface_hub pretrained weight download.")
else:
    # Download all files from the repository
    try:
        snapshot_download(repo_id=model_name, local_dir=MODEL_CP_PATH)
        print(f"Model weights and files downloaded to: {MODEL_CP_PATH}")
    except Exception as e:
        print(f"Please request Llama-3.2-1B via https://huggingface.co/meta-llama/Llama-3.2-1B and run `huggingface-cli login`.")

# config = model.ModelConfig.llama3_2_1b()  # pick correponding config based on model version
# config = model.ModelConfig.llama3_8b()  # pick correponding config based on model version
config = model.ModelConfig.llama3_2_3b_instruct()  # pick correponding config based on model version

llama3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)

sys.exit("Error message")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)

def templatize(prompts):
    out = []
    for p in prompts:
        out.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": p},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                # enable_thinking=True,
            )
        )
    return out

inputs = templatize(
    [
        "which is larger 9.9 or 9.11?",
        "Why is the sky blue?",
        "How do you say cheese in French?",
    ]
)

sampler = sampler.Sampler(llama3, tokenizer, sampler.CacheConfig(cache_size=256, num_layers=28, num_kv_heads=8, head_dim=128))

# --- Benchmark Start ---
start_time = time.perf_counter()
out = sampler(inputs, total_generation_steps=128, echo=True)
end_time = time.perf_counter()
# --- Benchmark End ---

for t in out.text:
    print(t)
    print('*' * 30)

# Print the benchmark result
print(f"\nBenchmark: Text generation completed in {end_time - start_time:.4f} seconds.")