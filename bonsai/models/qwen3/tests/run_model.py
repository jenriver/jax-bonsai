from bonsai.models.qwen3 import params
from bonsai.models.qwen3 import model
from flax import nnx
from huggingface_hub import snapshot_download
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from bonsai.generate import sampler, inference, utils
import jax
import jax.numpy as jnp
import numpy as np


model_name = 'Qwen/Qwen3-0.6B'
MODEL_CP_PATH = '/tmp/models-bonsai/' + model_name.split('/')[1]
if not os.path.isdir(MODEL_CP_PATH):
  snapshot_download(model_name, local_dir=MODEL_CP_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)

def tokenize(tokenizer, input: list[str], pad_idx: int = 0):
    lines = [tokenizer.apply_chat_template([{"role": "user", "content": line}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for line in input]
    lines = [tokenizer.encode(line, add_special_tokens=False) for line in lines]
    max_len = utils.next_power_of_2(max(len(line) for line in lines))
    output = [[pad_idx] * (max_len - len(line)) + line for line in lines]
    # JIYOUNHA: use utils.pad_to_length instead?
    return np.array(output)

# tokens: [[151644    872    198  10234    525   1052   3040  15584    304  71687
#     1380    279   8380  38835    926   4122     30 151645    198 151644
#    77091    198]
#  [     0      0      0      0      0      0      0      0      0      0
#        0 151644    872    198  23085   1616     30 151645    198 151644
#    77091    198]]
tokens = jnp.array(
    tokenize(
        tokenizer,
        [
            "Why is the sky blue instead of any other color like purple?",
            "Who is Dwight Schrute?"
        ],
    )
)
print(f'tokens shape: {tokens.shape}')

# cache: [<bonsai.generate.inference.LayerCache object at 0x7fbcfa90b8c0>, <bonsai.generate.inference.LayerCache object at 0x7fbc85112710>,
cache = inference.init_cache(num_layers=28, cache_size=256, batch_size=tokens.shape[0], num_kv_heads=8, head_dim=128)
# cache_state: {}
cache_state = nnx.to_pure_dict(nnx.state(cache))

# model_config: ModelConfig(num_layers=28, vocab_size=151936, embed_dim=1024, hidden_dim=3072, num_heads=16, head_dim=128, num_kv_heads=8, rope_theta=1000000, norm_eps=1e-06, shd_config=ShardingConfig(emb_vd=('tp', 'fsdp'), emb_dv=('fsdp', 'tp'), q_weight_ndh=('tp', 'fsdp', None), kv_weight_ndh=('tp', 'fsdp', None), o_weight_nhd=('tp', None, 'fsdp'), ffw_weight_df=('fsdp', 'tp'), ffw_weight_fd=('tp', 'fsdp'), rms_norm_weight=('tp',), act_btd=('fsdp', None, 'tp'), act_btf=('fsdp', None, 'tp'), act_btnh=('fsdp', None, 'tp', None)))
config = model.ModelConfig.qwen3_0_6b()  # pick correponding config based on model version
# model: Qwen3( # Param: 751,632,384 (1.5 GB)
#   embedder=Embedder( # Param: 155,582,464 (311.2 MB)
#     input_embedding=Param( # 155,582,464 (311.2 MB)
#       value=Array(shape=(151936, 1024), dtype=dtype(bfloat16)),
#       sharding="('tp', 'fsdp')"
#     ),
model = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)

# model_state: {'embedder': {'input_embedding': Array([[-0.00927734, 0.0336914, -0.074707, ..., 0.0119629, -0.0105591, 0.0159912],
# dict_keys(['embedder', 'final_norm', 'layers', 'lm_head'])
model_state = jax.device_put(nnx.to_pure_dict(nnx.state(model)), jax.devices("cuda")[0])

sampler = sampler.Sampler(model, tokenizer, sampler.KVCacheConfig(cache_size=256, num_layers=28, num_kv_heads=8, head_dim=128))

# warm up
print(f'Warming up XLA compiler...')
_ = sampler(tokens, total_generation_steps=128, max_prompt_length=64, temperature=0.6, top_p=0.95, echo=True)

start_time = time.time()

with jax.profiler.trace("/tmp/profile-data"):
    out = sampler(tokens, total_generation_steps=128, max_prompt_length=64, temperature=0.6, top_p=0.95, echo=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generated response in {elapsed_time:.2f} seconds")

    for t in out.text:
        print(t)
        print('*' * 30)
