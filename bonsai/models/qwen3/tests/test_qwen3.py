# This file should be run as a module from the project root using:
# python -m bonsai.models.qwen3.tests.test_qwen3

import os
import time

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.generate import sampler
from bonsai.models.qwen3 import model, params

model_name = "Qwen/Qwen3-0.6B"

MODEL_CP_PATH = "/tmp/qwen3-0.6b-weights"  # Specify your desired download directory

if os.path.isdir(MODEL_CP_PATH):
    print(f"'{MODEL_CP_PATH}' exists, skipping huggingface_hub pretrained weight download.")
else:
    # Download all files from the repository
    snapshot_download(repo_id=model_name, local_dir=MODEL_CP_PATH)
    print(f"Model weights and files downloaded to: {MODEL_CP_PATH}")

config = model.ModelConfig.qwen3_0_6_b()  # pick correponding config based on model version
qwen3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)

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

sampler = sampler.Sampler(
    qwen3, tokenizer, sampler.CacheConfig(cache_size=256, num_layers=28, num_kv_heads=8, head_dim=128)
)

# Help precompile
out = sampler(inputs, total_generation_steps=128, echo=True)
# --- Benchmark Start ---
start_time = time.perf_counter()
out = sampler(inputs, total_generation_steps=128, echo=True)
end_time = time.perf_counter()
# --- Benchmark End ---

for t in out.text:
    print(t)
    print("*" * 30)

# Print the benchmark result
print(f"\nBenchmark: Text generation completed in {end_time - start_time:.4f} seconds.")
