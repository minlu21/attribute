#%%
from transformer_lens import HookedTransformer
model_tl = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.2-1B")
#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map={"": "cuda"})
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#%%
import torch
torch.set_grad_enabled(False)
tokenized = tokenizer("Michael Jordan plays the sport of", return_tensors="pt").to("cuda")
#%%
cache = model_tl.run_with_cache(tokenized.input_ids)
#%%
def record_hook(self, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    self.saved_output = output
    self.saved_input = input
#%%
from collections import OrderedDict
for m in model_hf.modules():
    m._forward_hooks = OrderedDict()
for layer in range(16):
    model_hf.model.layers[layer].mlp.register_forward_hook(record_hook)
    model_hf.model.layers[layer].post_attention_layernorm.register_forward_hook(record_hook)
out = model_hf(tokenized.input_ids)
#%%
import numpy as np
import torch
for layer in range(16):
    # hid_1 = cache[1][f"blocks.{layer}.hook_mlp_out"][:, 1:]
    body = cache[1][f"blocks.{layer}.hook_resid_mid"]
    hid_2 = model_hf.model.layers[layer].post_attention_layernorm.saved_input[:, 1:]
    # torch.save(body, f"results/mlp_in_{layer}.pt")
    torch.save(hid_2, f"results/mlp_in_{layer}.pt")
    hid_1 = body[:, 1:]

    # hid_2 = model_hf.model.layers[layer].mlp.saved_input[:, 1:]

    # hid_2 = model_hf.model.layers[layer].mlp.saved_output[:, 1:]
    xs = hid_1.flatten().cpu().float().numpy()
    ys = hid_2.flatten().cpu().float().numpy()
    correlation = np.corrcoef(xs, ys)[0, 1] ** 2
    print(f"Layer {layer}: {correlation}")
#%%
