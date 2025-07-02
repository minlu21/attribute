#%%
%env CUDA_VISIBLE_DEVICES=6
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
#%%
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."# "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
tokenized = tokenizer(text, return_tensors="pt").to("cuda")
# %%
import torch
from sparsify import SparseCoder
from pathlib import Path
torch.set_grad_enabled(False)
coders = {}
base_dir = Path("/mnt/ssd-1/nev/e2e/checkpoints/gpt2-sweep/bs8-lr2e-4-none-ef128-k16/")
for submodule in base_dir.glob("*"):
    if not submodule.is_dir():
        continue
    coders[submodule.name] = SparseCoder.load_from_disk(submodule, device="cuda")
#%%
from collections import OrderedDict
from functools import partial
def code(module, input, output, name, coder):
    if isinstance(input, tuple):
        input = input[0]
    input = input.view(-1, input.shape[-1])
    output = output.view(-1, output.shape[-1])
    result = coder(input, output)
    print(name, result.fvu)
for m in model.modules():
    m._forward_hooks = OrderedDict()
for name, coder in coders.items():
    model.transformer.get_submodule(name).register_forward_hook(partial(code, name=name, coder=coder))
model(tokenized.input_ids);
#%%
