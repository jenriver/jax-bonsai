# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma3 model parameters.

This provides a mapping from the upstream checkpoints[1] to our implementation.

[1] https://github.com/google-deepmind/gemma
"""

from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
import re
import safetensors.flax as safetensors
from bonsai.models.gemma3 import model as model_lib
import sentencepiece as spm

# Pretrained
GEMMA3_1B_PT = 'gs://gemma-data/checkpoints/gemma3-1b-pt'
GEMMA3_4B_PT = 'gs://gemma-data/checkpoints/gemma3-4b-pt'
GEMMA3_12B_PT = 'gs://gemma-data/checkpoints/gemma3-12b-pt'
GEMMA3_27B_PT = 'gs://gemma-data/checkpoints/gemma3-27b-pt'
# Instruction Tuned
GEMMA3_1B_IT = 'gs://gemma-data/checkpoints/gemma3-1b-it'
GEMMA3_4B_IT = 'gs://gemma-data/checkpoints/gemma3-4b-it'
GEMMA3_12B_IT = 'gs://gemma-data/checkpoints/gemma3-12b-it'
GEMMA3_27B_IT = 'gs://gemma-data/checkpoints/gemma3-27b-it'
# Tokenizer
GEMMA3_TOKENIZER = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Gemma3:
  """Load a Gemma3 model from a checkpoint."""
  abs_model = nnx.eval_shape(
      lambda: model_lib.Gemma3(model_config, rngs=nnx.Rngs(0))
  )
  params = ocp.StandardCheckpointer().restore(checkpoint_path)
  params = _map_from_upstream_checkpoint(params)
  if mesh is not None:
    params = jax.tree.map(
        lambda x, shd: jnp.asarray(x, device=shd),
        params,
        nnx.to_pure_dict(nnx.get_named_sharding(nnx.state(abs_model), mesh)),
    )
  else:
    params = jax.tree.map(jnp.asarray, params)
  nnx.update(abs_model, params)
  return abs_model


PROMPT_TEMPLATE = """\
<start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
"""


def create_tokenizer(
    path: str = GEMMA3_TOKENIZER,
) -> spm.SentencePieceProcessor:
  spm_processor = spm.SentencePieceProcessor()
  model_proto = epath.Path(path).read_bytes()
  spm_processor.LoadFromSerializedProto(model_proto)
  return spm_processor


def _map_from_upstream_checkpoint(params):
  """Map from upstream checkpoint to our implementation."""
  # From:
  #
  # ('transformer/embedder', 'input_embedding') (262144, 1152)
  # ('transformer/final_norm', 'scale') (1152,)
  # ('transformer/layer_0/attn/_key_norm', 'scale') (256,)
  # ('transformer/layer_0/attn/_query_norm', 'scale') (256,)
  # ('transformer/layer_0/attn/attn_vec_einsum', 'w') (4, 256, 1152)
  # ('transformer/layer_0/attn/kv_einsum', 'w') (2, 1, 1152, 256)
  # ('transformer/layer_0/attn/q_einsum', 'w') (4, 1152, 256)
  # ('transformer/layer_0/mlp/gating_einsum', 'w') (2, 6912, 1152)
  # ('transformer/layer_0/mlp/linear', 'w') (6912, 1152)
  # ('transformer/layer_0/post_attention_norm', 'scale') (1152,)
  # ('transformer/layer_0/post_ffw_norm', 'scale') (1152,)
  # ('transformer/layer_0/pre_attention_norm', 'scale') (1152,)
  # ('transformer/layer_0/pre_ffw_norm', 'scale') (1152,)
  #
  # To:
  #
  # ('embedder', 'input_embedding') (262144, 1152)
  # ('final_norm', 'scale') (1152,)
  # ('layers', 0, 'attn', '_key_norm', 'scale') (256,)
  # ('layers', 0, 'attn', '_query_norm', 'scale') (256,)
  # ('layers', 0, 'attn', 'attn_vec_einsum', 'w') (4, 256, 1152)
  # ('layers', 0, 'attn', 'kv_einsum', 'w') (2, 1, 1152, 256)
  # ('layers', 0, 'attn', 'q_einsum', 'w') (4, 1152, 256)
  # ('layers', 0, 'mlp', 'down_proj', 'kernel') (6912, 1152)
  # ('layers', 0, 'mlp', 'gate_proj', 'kernel') (1152, 6912)
  # ('layers', 0, 'mlp', 'up_proj', 'kernel') (1152, 6912)
  # ('layers', 0, 'post_attn_norm', 'scale') (1152,)
  # ('layers', 0, 'post_ffw_norm', 'scale') (1152,)
  # ('layers', 0, 'pre_attention_norm', 'scale') (1152,)
  # ('layers', 0, 'pre_ffw_norm', 'scale') (1152,)
  new_params = {}
  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    module_path, param_name = key_path
    module_path = module_path.split('/')[1:]  # Remove the leading 'transformer'
    if module_path[0] == 'siglip_encoder':
      continue  # We don't support MM input yet.
    if module_path[0] == 'embedder':
      if len(module_path) > 1 and module_path[1].startswith('mm_'):
        continue  # We don't support MM input yet.
    if module_path[0] in ('embedder', 'final_norm'):
      new_params[(module_path[0], param_name)] = value
      continue
    # module_path should now look like ('layer_0', 'attn', '_key_norm')
    layer_idx = ('layers', int(module_path[0].removeprefix('layer_')))
    if module_path[1:] == ['mlp', 'gating_einsum']:
      new_params[(*layer_idx, 'mlp', 'gate_proj', 'kernel')] = value[0].T
      new_params[(*layer_idx, 'mlp', 'up_proj', 'kernel')] = value[1].T
    elif module_path[1:] == ['mlp', 'linear']:
      new_params[(*layer_idx, 'mlp', 'down_proj', 'kernel')] = value
    else:
      new_params[(*layer_idx, *module_path[1:], param_name)] = value
  return flax.traverse_util.unflatten_dict(new_params)





def generate_text_with_sentencepiece(prompt_text, max_length=50, sp_model=None, simulated_model=None, simulated_model_output_ids_for_test=None):
    """
    Generates text using a prompt, a SentencePiece model for tokenization,
    and a simulated language model.

    Args:
        prompt_text (str): The initial text to start generation.
        max_length (int): Maximum number of tokens to generate.
        sp_model (sentencepiece.SentencePieceProcessor): The loaded SentencePiece model.
        simulated_model (function): A function that takes token IDs and returns
                                    a prediction for the next token's probability distribution
                                    (or just the next token ID in our simple case).
    Returns:
        str: The generated text.
    """
    if sp_model is None:
        raise ValueError("SentencePiece model (sp_model) must be provided.")
    if simulated_model is None:
        raise ValueError("Simulated model (simulated_model) must be provided.")

    # Encode the prompt text into token IDs
    input_ids = sp_model.encode_as_ids(prompt_text)
    print(f"\nPrompt '{prompt_text}' encoded to IDs: {input_ids}")

    generated_ids = list(input_ids) # Start with the prompt IDs

    # Simulate token generation
    for _ in range(max_length):
        # In a real model, you'd pass `generated_ids` to your neural network
        # and get a probability distribution over the vocabulary for the next token.
        # Then you'd sample from that distribution (e.g., greedy, top-k, nucleus sampling).

        # Our simulated model simply returns the next ID from its fixed sequence
        # or a special "end of sequence" token if it runs out.
        if len(generated_ids) - len(input_ids) < len(simulated_model_output_ids_for_test):
            next_token_id = simulated_model_output_ids_for_test[len(generated_ids) - len(input_ids)]
        else:
            # Simulate an end-of-sequence token or just stop
            print("Simulated model reached end of its fixed output.")
            break

        generated_ids.append(next_token_id)
        # Optional: Print current generation state
        # print(f"  Generated ID: {next_token_id}, Current sequence: {sp_model.decode_ids(generated_ids)}")

    # Decode the complete sequence of generated IDs back to text
    full_generated_text = sp_model.decode_ids(generated_ids)
    return full_generated_text