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
from typing import Optional

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
import bonsai.generate.beam_search as beam_search_lib
import bonsai.generate.tokenizer_adapter as tok_adapter


class LayerCache:
    def __init__(self, batch_size, cache_size, num_kv_heaads, dtype):
        k_cache: nnx.Cache(jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim), dtype=dtype))
        v_cache: nnx.Cache(jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim), dtype=dtype))
Cache = list[LayerCache]

def init_cache(
    num_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.int32,
) -> Cache:
    return [LayerCache(batch_size, cache_size, num_kv_heads, dtype) for x in range(num_layers)]

def forward(
    tokens: jax.Array | None = None,
    input_embeds: jax.Array | None = None,
    per_layer_inputs: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    *,
    model_state: nnx.State,
    cache_state: nnx.State | None = None,
    cfg: Config,
):
    return [0], [0], cache_state


def compute_positions_from_segment_ids(segment_ids: jax.Array):
    assert segment_ids.ndim == 2

    def fn_(carry, segment_id):
        prev_segment_id, counter = carry
        next_counter = jnp.where(segment_id == prev_segment_id, counter + 1, 0)
        return (segment_id, next_counter), next_counter

    b, segment_ids = segment_ids.shape[0], segment_ids.astype(jnp.int32)
    return jax.lax.scan(fn_, ((2**30) * jnp.ones(b, jnp.int32), jnp.zeros(b, jnp.int32)), segment_ids.T)[1].T


def compute_rope_embeddings(
    dim: int, positions: jax.Array, theta: float, scaling_factor: float
) -> tuple[jax.Array, jax.Array]:
    factor = 1.0 / scaling_factor
    inv_freq = factor / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    input = jnp.einsum("BT,k->BTk", positions, inv_freq, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(input), jnp.cos(input)


@property
def transformer(model) -> nnx.Module:
    model_state = nnx.variables(transformer)
    flattened_model_state = jax.tree.leaves(
        model_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    transformer_graphdef = nnx.graphdef(transformer)
    return nnx.merge(transformer_graphdef, flattened_model_state)

def prefill(
    model: nnx.Module,
    tokens: jax.Array | None = None,
    input_embeds: jax.Array | None = None,
    per_layer_inputs: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    *,
    model_state: nnx.State,
    cache_state: nnx.State | None = None,
    cfg: Config,
    pad_id: int = 0,
) -> (jax.Array, jax.Array, nnx.State):

    print(f'tokens: {tokens}')
    print(f'tokens padded len : {len(tokens)}, padded: {tokens}')
    tokens_mask = tokens != pad_id
    segment_ids = jax.jit(jnp.where)(tokens_mask, 1, 0)
    print(f'tokens_mask: {tokens_mask}')
    print(f'segment_ids: {segment_ids}')

    ###################### compute positional embeddings.
    positions = compute_positions_from_segment_ids(segment_ids)
    print(f'positions: {positions}')
    print(f'cache_state: {cache_state}')
    if cache_state != {}:
        batch_size, cache_seq_len = tokens.shape[0], cache_state[0]["k"]["entry"].shape[-2]
        cache = nnx.eval_shape(lambda: init_cache(batch_size, cache_seq_len, cfg))
        nnx.update(cache, cache_state)
    else:
        cache = [None for _ in range(cfg.num_layers)]
    if cache_state != {}:
        positions = (positions + cache[0].fill()[:, None]) % cache[0].size
    sin_global, cos_global = compute_rope_embeddings(cfg.head_dim, positions, cfg.rope_theta, cfg.rope_scaling_factor)
    sin_local, cos_local = compute_rope_embeddings(cfg.head_dim, positions, cfg.local_rope_theta, 1.0)


    print(f'sin_global: {sin_global}')
    print(f'cos_global: {cos_global}')
    print(f'sin_local: {sin_local}')
    print(f'cos_local: {cos_local}')

    print(f'sin_global.shape: {sin_global.shape}')
    print(f'cos_global.shape: {cos_global.shape}')
    print(f'sin_local.shape: {sin_local.shape}')
    print(f'cos_local.shape: {cos_local.shape}')
    # sin_global.shape: (2, 32, 64)
    # cos_global.shape: (2, 32, 64)
    # sin_local.shape: (2, 32, 64)
    # cos_local.shape: (2, 32, 64)

    ###################### compute per-layer input
    

    return ([0], [0], cache_state)