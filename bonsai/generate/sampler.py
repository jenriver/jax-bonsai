# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vanilla sampler for LLM generation."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
from bonsai.generate import utils
from functools import partial

LayerCache = dict[str, jaxtyping.Array]
Cache = list[str, LayerCache]


@jax.jit
def prefill(
    graphdef, state, sampler_state: _SamplerState
) -> _SamplerState:
  sampler: Sampler = nnx.merge(graphdef, state)

  batch_size = sampler_state.token_buffer.shape[0]
  tokens = jax.lax.dynamic_slice(
      sampler_state.token_buffer,
      start_indices=jnp.zeros(
          (sampler_state.token_buffer.ndim,), dtype=jnp.int32
      ),
      slice_sizes=(batch_size, sampler_state.num_input_tokens),
  )
  step_positions = jax.lax.dynamic_slice(
      sampler_state.positions,
      start_indices=jnp.zeros(
          (sampler_state.token_buffer.ndim,), dtype=jnp.int32
      ),
      slice_sizes=(batch_size, sampler_state.num_input_tokens),
  )

  input_mask = tokens != sampler.tokenizer.pad_token_id
  attention_mask = utils.make_causal_attn_mask(
      input_mask, sampler.cache_config.cache_size
  )

  logits, cache = sampler.transformer(
      tokens,
      step_positions,
      sampler_state.cache,
      attention_mask,
  )
  token_buffer = sampler_state.token_buffer
  done = sampler_state.done
  positions = sampler_state.positions
  if sampler_state.logits_buffer is not None:
    logits_buffer = jax.lax.dynamic_update_slice(
        sampler_state.logits_buffer,
        logits.astype(sampler_state.logits_buffer.dtype),
        (0, 1, 0),
    )
  else:
    logits_buffer = sampler_state.logits_buffer

  updated_sampler_state = _SamplerState(
      decoding_step=sampler_state.decoding_step,
      num_input_tokens=sampler_state.num_input_tokens,
      token_buffer=token_buffer,
      positions=positions,
      logits_buffer=logits_buffer,
      cache=cache,
      done=done,
      total_sampling_steps=sampler_state.total_sampling_steps,
      temperature=sampler_state.temperature,
      sampling_parameters=sampler_state.sampling_parameters,
      seed=sampler_state.seed,
      sampling_mode=sampler_state.sampling_mode,
  )
  updated_sampler_state = sample(
      logits=logits,
      cache=cache,
      eos=sampler.tokenizer.eos_token_id,
      sampler_state=updated_sampler_state,
  )
  return updated_sampler_state

@jax.jit
def decode(
    graphdef, state, sampler_state: _SamplerState
) -> _SamplerState:
  """Internal generating function (to be jitted)."""
  sampler: Sampler = nnx.merge(graphdef, state)

  def sample_with_params(sampler_state: _SamplerState):
    return sample_step(graphdef, state, sampler_state)

  def cond_fn(sampler_state: _SamplerState):
    return (
        sampler_state.decoding_step < sampler_state.total_sampling_steps
    ) & jnp.any(jnp.logical_not(sampler_state.done))

  return jax.lax.while_loop(cond_fn, sample_with_params, sampler_state)

@jax.jit
def sample_step(
    graphdef, state, sampler_state: _SamplerState
) -> _SamplerState:
  """Performs a single sampling step."""
  sampler: Sampler = nnx.merge(graphdef, state)
  batch_size = sampler_state.token_buffer.shape[0]
  decoding_step = sampler_state.decoding_step

  last_token = sampler_state.token_buffer[:, decoding_step]
  last_token = last_token.reshape((batch_size, 1))
  step_positions = jnp.expand_dims(
      sampler_state.positions[:, decoding_step], -1
  )

  input_mask = sampler_state.token_buffer == sampler.tokenizer.pad_token_id
  attention_mask = utils.compute_attention_masks(
      decoding_step, sampler.cache_config.cache_size, input_mask
  )

  logits, cache = sampler.transformer(
      last_token,
      step_positions,
      sampler_state.cache,
      attention_mask,
  )
  updated_sampler_state = sample(
      logits=logits,
      cache=cache,
      eos=sampler.tokenizer.eos_token_id,
      sampler_state=sampler_state,
  )

  if updated_sampler_state.logits_buffer is not None:
    next_logits = jnp.squeeze(logits, 1)
    logits_buffer = updated_sampler_state.logits_buffer.at[
        :, decoding_step + 1
    ].set(next_logits)
  else:
    logits_buffer = None

  updated_sampler_state = dataclasses.replace(
      updated_sampler_state,
      logits_buffer=logits_buffer,
  )
  return updated_sampler_state

@jax.jit
def sample(
    logits: jnp.ndarray,
    eos: int,
    cache: Cache,
    sampler_state: _SamplerState,
) -> _SamplerState:
  """Samples a token from the logits."""

  logits = logits[:, -1][:, None, :]  # B, 1, V
  decoding_step = sampler_state.decoding_step
  token_buffer = sampler_state.token_buffer
  done = sampler_state.done
  logits_buffer = sampler_state.logits_buffer

  if sampler_state.sampling_mode == 'greedy':
    next_token_candidate = sample_best(logits)
  elif sampler_state.sampling_mode == 'top_p':
    key = jax.random.fold_in(sampler_state.seed, decoding_step)
    next_token_candidate = sample_top_p(
        logits,
        key,
        sampler_state.temperature,
        sampler_state.sampling_parameters['top_p'],
        sampler_state.sampling_parameters['top_k'],
    )
  else:
    raise ValueError(
        'Unsupported sampling mode: %s' % sampler_state.sampling_mode
    )
  token_buffer = token_buffer.at[:, decoding_step + 1].set(
      next_token_candidate
  )

  done = done | jnp.equal(token_buffer[:, decoding_step + 1], eos)
  return _SamplerState(
      decoding_step=sampler_state.decoding_step + 1,
      num_input_tokens=sampler_state.num_input_tokens,
      token_buffer=token_buffer,
      positions=sampler_state.positions,
      logits_buffer=logits_buffer,
      cache=cache,
      done=done,
      total_sampling_steps=sampler_state.total_sampling_steps,
      temperature=sampler_state.temperature,
      sampling_parameters=sampler_state.sampling_parameters,
      seed=sampler_state.seed,
      sampling_mode=sampler_state.sampling_mode,
  )

@flax.struct.dataclass
class _SamplerState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: Cache

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # Random seed for sampling.
  seed: jax.Array

  # The sampling mode to use, one of "greedy", "top_p" or "beam_search"
  sampling_mode: str = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Sampling parameters.
  # For top_p, it contains "top_p" and "top_k".
  # For beam search, it contains "beam_size"
  sampling_parameters: dict[str, float | int] = flax.struct.field(
      pytree_node=False
  )

@dataclasses.dataclass
class SamplerOutput:
  """Output of the sampler."""

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[jax.Array] | jax.Array

  # Tokens corresponding to the generated samples.
  tokens: list[jax.Array] | jax.Array

  # Left padded prompt tokens.
  padded_prompt_tokens: jax.Array


@dataclasses.dataclass(frozen=True)
class KVCacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


def _sample_top_p(
    probs: jnp.ndarray, p: float, key: jax.Array, k: int | None = None
) -> jnp.ndarray:
  """Sample a token using top-p sampling."""
  k = probs.shape[-1] if k is None else k
  probs_sorted, indices = jax.lax.top_k(probs, k=k)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > p
  probs_sorted = jnp.where(mask, 0.0, probs_sorted)
  probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

  next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

  next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)
  return next_token


def sample_top_p(
    logits, key, temperature: float, top_p: float, top_k: int | None
):
  probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
  next_token = _sample_top_p(probs, top_p, key, top_k)
  return next_token


def sample_best(logits):
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  return next_token

def init_cache(
    n_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> Cache:
  """Create KV cache for the transformer.

  Args:
    n_layers: The number of attention layers.
    cache_size: The size of the cache.
    batch_size: The batch size.
    num_kv_heads: The number of KV attention heads.
    head_dim: The dimension of the KV attention head.
    dtype: The data type of the cache.

  Returns:
    The KV cache for one attention block.
  """

  def _init_layer_cache() -> LayerCache:
    return {
        'k': jnp.zeros(
            (batch_size, cache_size, num_kv_heads, head_dim), dtype=dtype
        ),
        'v': jnp.zeros(
            (batch_size, cache_size, num_kv_heads, head_dim), dtype=dtype
        ),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
    }

  cache = [_init_layer_cache() for i in range(n_layers)]
  return cache

class Sampler(nnx.Module):
  """Sampler for transformer model."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: KVCacheConfig,
  ):
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      cache_config: configuration for the KV cache.
    """
    self.tokenizer = tokenizer
    self.cache_config = cache_config
    self.transformer = transformer

  @property
  def dtype(self) -> jnp.dtype:
    return jnp.bfloat16

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
      temperature: float,
      top_p: float | None,
      top_k: int | None,
      seed: jax.Array,
  ) -> _SamplerState:
    """Initializes the sampling state given input prompts."""
    batch_size = all_input_ids.shape[0]
    num_input_tokens = all_input_ids.shape[1]
    buffer_size = total_sampling_steps + 1

    token_buffer = jnp.full(
        (
            batch_size,
            buffer_size,
        ),
        self.tokenizer.pad_token_id,
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    input_mask = input_mask.at[:, :num_input_tokens].set(
        all_input_ids != self.tokenizer.pad_token_id
    )
    positions = utils.build_positions_from_mask(input_mask)

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    cache = init_cache(
        n_layers=self.cache_config.num_layers,
        cache_size=self.cache_config.cache_size,
        batch_size=batch_size,
        num_kv_heads=self.cache_config.num_kv_heads,
        head_dim=self.cache_config.head_dim,
        dtype=self.dtype,
    )

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, buffer_size, self.transformer.num_embed),
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None
    sampling_parameters = {}
    sampling_mode = [None]

    if top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')
      sampling_parameters['top_p'] = top_p
      sampling_parameters['top_k'] = top_k

    if sampling_mode[0] is None:
      sampling_mode[0] = 'greedy'

    logging.debug('Using sampling mode: %s', sampling_mode[0])

    return _SamplerState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=total_sampling_steps,
        temperature=temperature,
        sampling_parameters=sampling_parameters,
        seed=seed,
        sampling_mode=sampling_mode[0],
    )

  def tokenize(self, input_string: str) -> jax.Array:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else []
    input_ids = jnp.array(bos_tok + input_ids, dtype=jnp.int32)
    return input_ids

  def __call__(
      self,
      tokens: jnp.array,
      total_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      seed: jax.Array | None = None,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    If top_p is provided, the sampling mode will be top_p.
    If None of them are provided, the sampling mode will be greedy.

    Args:
      tokens: input tokens to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      max_prompt_length: maximum length of the prompt. Specify to avoid
        recompilation on different prompt lengths.
      echo: whgether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      top_k: top-k sampling threshold.
      seed: random seed for sampling.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    
    if seed is None:
      seed = jax.random.PRNGKey(0)
    sampler_state = self.init_sample_state(
        tokens,
        include_logits=return_logits, 
        total_sampling_steps=total_generation_steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    # Deconstruct the sampler into pytress passable as jax arguments.
    graphdef, state = nnx.split(self)
    sampler_state = prefill(graphdef, state, sampler_state)
    sampler_state = decode(graphdef, state, sampler_state)
    token_buffers = sampler_state.token_buffer

    out_tokens = []
    for i, token_buffer in enumerate(token_buffers):
      start_idx = (
          utils.find_first_non_pad_idx(token_buffer, self.tokenizer.pad_token_id)
          if echo
          else max_prompt_length
      )
      end_idx = (
          utils.find_first_eos_idx(
              token_buffer[max_prompt_length:], self.tokenizer.eos_token_id
          )
          + max_prompt_length
      )
      out_tokens.append(token_buffer[start_idx:end_idx])

    decoded_outputs = [
        self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
    ]

    result = SamplerOutput(
        text=decoded_outputs,
        logits=[],
        tokens=out_tokens,
        padded_prompt_tokens=tokens,
    )
    return result