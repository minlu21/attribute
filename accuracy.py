#%%
%load_ext autoreload
%autoreload 2
from attribute.caching import TranscodedModel


model_name = "roneneldan/TinyStories-33M"
transcoder_path = "../e2e/checkpoints/clt-ts/ts"

model = TranscodedModel(
    model_name=model_name,
    transcoder_path=transcoder_path,
    device="cuda",
)
#%%
prompt = "When John and Mary went to the store, John gave a bag to"
og_activations = model(prompt)
# %%
from collections import defaultdict

def collect_values(activations):
    values = {}
    for l, o in activations.mlp_outputs.items():
        act, loc = o.activation, o.location
        for seq in range(loc.shape[1]):
            for f, a in zip(loc[0, seq].tolist(), act[0, seq].tolist()):
                values[(seq, l, f)] = a
    return values

per_layer = defaultdict(set)
for l, o in og_activations.mlp_outputs.items():
    act, loc = o.activation, o.location
    per_layer[l].update(loc.flatten().unique().tolist())
per_layer = {k: sorted(v) for k, v in per_layer.items()}
og_values = collect_values(og_activations)
#%%
import random
remove_n = 1
sample_removed = {k: random.sample(v, remove_n) for k, v in per_layer.items()}
masked_activations = model(prompt, mask_features=sample_removed)
masked_values = collect_values(masked_activations)
for k, v in og_values.items():
    if k in masked_values:
        print(k, v, masked_values[k])
