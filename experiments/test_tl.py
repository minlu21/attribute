#%%
from transformer_lens import HookedTransformer
model_tl = HookedTransformer.from_pretrained_no_processing("google/gemma-2-2b")
model_tl.requires_grad_(False)
#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
model_hf = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map={"": "cuda"})
model_hf.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
#%%
import torch
tokenized = tokenizer(
    # "Michael Jordan plays the sport of"
    "In another moment, down went Alice after it, never once considering how in the world she was to get out again."
    , return_tensors="pt").to("cuda")
#%%
from functools import partial
def hook_grad(output, hook):
    output = output.clone()
    output.requires_grad_(True)
    return output
cache = {}
def record_hook(output, hook, name):
    output.retain_grad()
    cache[name] = output
    return output
record_hooks = [
    (f"blocks.{i}.hook_resid_pre", partial(record_hook, name=f"blocks.{i}.hook_resid_pre")) for i in range(16)
] + [
    (f"blocks.{i}.hook_resid_mid", partial(record_hook, name=f"blocks.{i}.hook_resid_mid")) for i in range(16)
]
# with model_tl.hooks([("blocks.0.hook_resid_pre", hook_grad)] + record_hooks):
    # model_tl(tokenized.input_ids)
_, cache = model_tl.run_with_cache(tokenized.input_ids)
def record_hook(self, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    # output.retain_grad()
    # self.saved_output = output
    # output = output.clone()
    # output.requires_grad_(True)
    output.retain_grad()
    self.saved_output = output
    return output
def record_pre_hook(self, input):
    if isinstance(input, tuple):
        input = input[0]
    # input = input.clone()
    # input.requires_grad_(True)
    input.retain_grad()
    self.saved_input = input
    return input
from collections import OrderedDict
for m in model_hf.modules():
    m._forward_hooks = OrderedDict()
    m._forward_pre_hooks = OrderedDict()
for layer in range(26):
    model_hf.model.layers[layer].mlp.register_forward_hook(record_hook)
    model_hf.model.layers[layer].mlp.register_forward_pre_hook(record_pre_hook)
    model_hf.model.layers[layer].post_attention_layernorm.register_forward_hook(record_hook)
    model_hf.model.layers[layer].post_attention_layernorm.register_forward_pre_hook(record_pre_hook)
    model_hf.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(record_hook)
    model_hf.model.layers[layer].pre_feedforward_layernorm.register_forward_pre_hook(record_pre_hook)
def set_grad_hook(self, input, output):
    output = output.clone()
    output.requires_grad_(True)
    return output
model_hf.model.embed_tokens.register_forward_hook(set_grad_hook)
out = model_hf(tokenized.input_ids)
import numpy as np
import torch

# cache["blocks.15.hook_resid_mid"].sum().backward()
# model_hf.model.layers[15].post_attention_layernorm.saved_input.sum().backward()

for layer in range(30):
    hid_1 = cache[f"blocks.{layer}.ln2.hook_normalized"][:, 1:]
    # hid_1 = cache[f"blocks.{layer}.attn.hook_z"][:, 1:]
    # body = cache[f"blocks.{layer}.hook_resid_mid"].grad
    hid_2 = model_hf.model.layers[layer].pre_feedforward_layernorm.saved_output[:, 1:]
    hid_2 /= (1 + model_hf.model.layers[layer].pre_feedforward_layernorm.weight)
    with torch.no_grad():
        # hid_2 = model_hf.model.layers[layer].post_attention_layernorm.saved_input.grad[:, 1:]
        # torch.save(body, f"results/mlp_in_{layer}.pt")
        torch.save(hid_2, f"results/mlp_in_{layer}.pt")
        # hid_1 = body[:, 1:]

        # hid_2 = model_hf.model.layers[layer].mlp.saved_input[:, 1:]

        # hid_2 = model_hf.model.layers[layer].mlp.saved_output[:, 1:]
        xs = hid_1.flatten().cpu().float().numpy()
        ys = hid_2.flatten().cpu().float().numpy()
        correlation = np.corrcoef(xs, ys)[0, 1] ** 2
        print(f"Layer {layer}: {correlation}")
#%%
cache["blocks.0.attn.hook_z"]
#%%
model_tl
