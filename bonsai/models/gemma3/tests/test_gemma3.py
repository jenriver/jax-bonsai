# This file should be run as a module from the project root using:
# python -m bonsai.models.gemma3.tests.test_gemma3


from bonsai.models.gemma3 import model
from bonsai.models.gemma3 import params

import sentencepiece as spm

model_name = params.GEMMA3_1B_PT
# model_name = "google/gemma-3-1b-pt"

# gsutil cp -r gs://gemma-data/checkpoints/gemma3-1b-pt /tmp/
# gsutil cp -r gs://gemma-data/tokenizers/tokenizer_gemma3.model /tmp/
MODEL_CP_PATH = "/tmp/gemma3-1b-pt"  # Specify your desired download directory

"""
if os.path.isdir(MODEL_CP_PATH):
    print(f"'{MODEL_CP_PATH}' exists, skipping huggingface_hub pretrained weight download.")
else:
    # Download all files from the repository
    snapshot_download(repo_id=model_name, local_dir=MODEL_CP_PATH)
    print(f"Model weights and files downloaded to: {MODEL_CP_PATH}")
"""

config = model.ModelConfig.gemma3_1b()  # pick correponding config based on model version
# gemma3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)
gemma3 = params.create_model_from_checkpoint(MODEL_CP_PATH, config)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)
tokenizer = params.create_tokenizer(params.GEMMA3_TOKENIZER)


# 1. Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("/tmp/tokenizer_gemma3.model")
# print(f"SentencePiece model loaded: {sp.model_path()}")

# --- Simulate a simple language model ---
# In a real scenario, this would be a complex neural network.
# Here, we'll just have a predefined "output sequence" in terms of token IDs.
# Let's say our model, given the prompt "Hello world", would typically generate
# " This is a test."
# We'll get the token IDs for " This is a test." from our sp model.
target_continuation_text = "Lorem Ipsum This is a test."
simulated_model_output_ids_for_test = sp.encode_as_ids(target_continuation_text)
print(f"Simulated model's fixed output token IDs: {simulated_model_output_ids_for_test}")


# --- Run the generation ---
prompt = "Hello world"
generated_output = params.generate_text_with_sentencepiece(
    prompt_text=prompt,
    max_length=200,  # Limit generation to a reasonable number of tokens
    sp_model=sp,
    simulated_model=gemma3,
    simulated_model_output_ids_for_test=simulated_model_output_ids_for_test,
)

print("\n--- Generation Complete ---")
print(f"Prompt: '{prompt}'")
print(f"Generated Text: '{generated_output}'")
