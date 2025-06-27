from bonsai.models.gemma3 import params
from bonsai.models.gemma3 import model
from flax import nnx
import os
from huggingface_hub import snapshot_download
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from bonsai.generate import sampler


model_name = 'google/gemma-3-1b-it'
MODEL_CP_PATH = '/tmp/models-bonsai/' + model_name.split('/')[1]
if not os.path.isdir(MODEL_CP_PATH):
  snapshot_download(model_name, local_dir=MODEL_CP_PATH)

config = model.ModelConfig.gemma3_1b()  # pick correponding config based on model version
model = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)

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
          # enable_thinking=True,
          enable_thinking=False,
      )
    )
  return out

inputs = templatize(
    [
        "Why is the sky blue?",
    ]
)
inputs2 = templatize(["why am i a blue boy with a yellow sock?"])
inputs3 = templatize(["am i the baddest blueberry out there?"])

input_list = [inputs, inputs2, inputs3]

sampler = sampler.Sampler(model, tokenizer, sampler.KVCacheConfig(cache_size=256, num_layers=26, num_kv_heads=1, head_dim=256))

# warm up
print(f'Warming up XLA compiler...')
_ = sampler(inputs, total_generation_steps=128, echo=True)

for i in range(3):
        start_time = time.time()
        print(f'User: {input_list[i]}')
        out = sampler(input_list[i], total_generation_steps=128, echo=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Generated response in {elapsed_time:.2f} seconds")

        for t in out.text:
          print(t)
          print('*' * 30)
