#%%
from IPython import get_ipython
if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

from attribute.caching import TranscodedModel
import torch
torch.set_grad_enabled(False)

model = TranscodedModel(
    model_name="meta-llama/Llama-3.2-1B",
    transcoder_path="EleutherAI/skip-transcoder-Llama-3.2-1B-131k",
    device="cuda",
)
#%%
layer, feature = 11, 11719
prompt = "The meaning of the word X is \""
tokenized_prompt = model.tokenizer(prompt)
print([model.decode_token(token) for token in tokenized_prompt])
og_activations = model(prompt)
model(prompt, errors_from=og_activations, steer_features={})
#%%
