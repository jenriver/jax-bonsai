# This file should be run as a module from the project root using:
# python -m bonsai.models.qwen3.tests.test_model
import os
import time
import traceback
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.generate import sampler
from bonsai.models.qwen3 import model, params

# --- Configuration Constants ---
MODEL_CONFIGS = {
    "Qwen/Qwen3-0.6B": {"config_fn": model.ModelConfig.qwen3_0_6b},
    "Qwen/Qwen3-1.7B": {"config_fn": model.ModelConfig.qwen3_1_7b},
    "Qwen/Qwen3-14B": {"config_fn": model.ModelConfig.qwen3_14b},
}


# --- Model and Tokenizer Loading ---
def load_model_and_tokenizer(model_name: str, local_cache_path: Path) -> tuple[model.Qwen3, AutoTokenizer]:
    """
    Downloads model weights if necessary, then loads the model and tokenizer.
    """
    model_cache_path = os.path.join(local_cache_path, model_name.replace("/", "_"))
    if os.path.isdir(model_cache_path):
        print(f"'{model_cache_path}' exists, skipping Hugging Face Hub download.")
    else:
        print(f"Downloading model weights and files to: {model_cache_path}")
        snapshot_download(repo_id=model_name, local_dir=model_cache_path)
        print("Download complete.")

    config = MODEL_CONFIGS[model_name]["config_fn"]()
    model = params.create_model_from_safe_tensors(model_cache_path, config)
    tokenizer = AutoTokenizer.from_pretrained(model_cache_path)
    return model, tokenizer


# --- Prompt Templatization ---
def apply_chat_template_to_prompts(tokenizer: AutoTokenizer, prompts: List[str]) -> List[str]:
    """
    Applies the chat template to a list of user prompts.

    Args:
        tokenizer: The pre-trained AutoTokenizer.
        prompts: A list of raw string prompts from the user.

    Returns:
        A list of templatized prompts ready for model input.
    """
    templated_outputs = []
    for p in prompts:
        templated_outputs.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to True if "thinking" messages are desired
            )
        )
    return templated_outputs


# --- Main Execution Block ---
def test_model_generation(model_name: str, local_cache_path: Path = "/tmp/models/", show_output: bool = False):
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name, local_cache_path)

        # Prepare input prompts
        input_prompts = apply_chat_template_to_prompts(
            tokenizer,
            [
                "Why is the sky blue?",
            ],
        )

        # Initialize the Sampler
        # CacheConfig parameters often depend on specific model architecture details
        # (e.g., number of layers, KV heads, head dimension).
        model_sampler = sampler.Sampler(
            tokenizer,
            sampler.KVCacheConfig(cache_size=256, num_layers=28, num_kv_heads=8, head_dim=128),
            temperature=0.7,
            top_p=0.9,
        )

        generated_output = model_sampler(model, input_prompts, total_generation_steps=128, echo=True)
        if show_output:
            print("\n--- Generated Outputs ---")
            for t in generated_output.text:
                print(t)
                print("*" * 30)
        if any(generated_output.text):
            print(f"Test Passed: Generation successful for {model_name}.")
            return "✅ Runs"
        else:
            print(f"Test Failed: No text generated for {model_name}.")
            return "⛔️ Not supported (Consider creating an issue)"

    except Exception as e:
        print(f"Test Failed: An error occurred during inference for {model_name}: {e}.")
        traceback.print_exc()
        return "⛔️ Not supported (Consider fixing or creating an issue)"


if __name__ == "__main__":
    # --- Example Usage ---
    # Choose the Qwen3 model you want to test.
    # Ensure its configuration is present and accurate in the `MODEL_CONFIGS` dictionary.
    model_to_test = "Qwen/Qwen3-0.6B"  # Change this to "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B", etc.

    result = test_model_generation(model_to_test, show_output=True)
    print(f"\nOverall Test Result for {model_to_test}: {result}")
