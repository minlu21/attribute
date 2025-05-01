#%%
%load_ext autoreload
%autoreload 2
from delphi.latents import LatentDataset
from delphi.config import SamplerConfig, ConstructorConfig
from natsort import natsorted
import torch

cache_path = "results/transcoder_128x/latents"
module_latents = {
    "h.0.mlp": torch.tensor([34765])
}
ds = LatentDataset(
    cache_path,
    SamplerConfig(), ConstructorConfig(),
    modules=natsorted(module_latents.keys()),
    latents=module_latents,
)
#%%
async for feat in ds:
    feat.display()
#%%
