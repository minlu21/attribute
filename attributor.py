#%%
%load_ext autoreload
%autoreload 2
%env CUDA_VISIBLE_DEVICES=1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.sparse_coder import SparseCoder

from attribute.mlp_attribution import AttributionGraph

import fire

from delphi.latents import LatentDataset
from delphi.config import SamplerConfig, ConstructorConfig
from natsort import natsorted
from collections import defaultdict
from pathlib import Path
from attribute.nodes import OutputNode
from attribute.utils import cantor
from attribute.caching import TranscodedModel
import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
#%%
model_name = "HuggingFaceTB/SmolLM2-135M"
hookpoint_fn = lambda hookpoint: hookpoint.replace("model.layers.", "layers.")
transcoder_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/sparsify/checkpoints/single_128x"
model = TranscodedModel(
    model_name=model_name,
    transcoder_path=transcoder_path,
    hookpoint_fn=hookpoint_fn,
    device="cuda",
)
#%%
# prompt = "When John and Mary went to the store, John gave a bag to"
# prompt = "Fact: Michael Jordan plays the sport of"
# prompt = "3 + 5 ="
prompt = "The National Digital Analytics Group (N"
transcoded_outputs = model(prompt)
#%%
%load_ext autoreload
%autoreload 2
attribution_graph = AttributionGraph(model, transcoded_outputs)
%time attribution_graph.flow(2_000)
#%%
save_dir = Path("../attribution-graphs-frontend")
attribution_graph.save_graph(save_dir)
#%%
module_latents = defaultdict(list)
for selected_node in selected_nodes:
    node = attribution_graph.nodes[selected_node]
    if node.node_type == "IntermediateNode":
        l, f = int(node.layer_index), int(node.feature_index)
        module_latents[model.temp_hookpoints_mlp[l]].append(f)
module_latents = {k: torch.tensor(v) for k, v in module_latents.items()}
# %%
cache_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/attribution_graph/results/transcoder_128x/latents"
ds = LatentDataset(
    cache_path,
    SamplerConfig(), ConstructorConfig(),
    modules=natsorted(module_latents.keys()),
    latents=module_latents,
)

#%%
logit_weight = model.logit_weight
# %%
bar = tqdm(total=sum(map(len, module_latents.values())))
def process_feature(feature):
    layer_idx = int(feature.latent.module_name.split(".")[-2])
    feature_idx = feature.latent.latent_index
    index = cantor(layer_idx, feature_idx)

    feature_dir = save_dir / "features" / scan
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{index}.json"

    if feature_path.exists():
        examples_quantiles = json.loads(feature_path.read_text())["examples_quantiles"]
    else:
        examples_quantiles = defaultdict(list)
        for example in feature.train:
            examples_quantiles[example.quantile].append(dict(
                is_repeated_datapoint=False,
                train_token_index=len(example.tokens) - 1,
                tokens=[model.decode_token(i) for i in example.tokens.tolist()],
                tokens_acts_list=example.activations.tolist(),
            ))
        examples_quantiles = [
            dict(
                quantile_name=f"Quantile {i}",
                examples=examples_quantiles[i],
            ) for i in sorted(examples_quantiles.keys())
        ]
    with torch.no_grad(), torch.autocast("cuda"):
        dec_weight = model.w_dec(layer_idx)[feature_idx]
        logits = logit_weight @ dec_weight
        top_logits = logits.topk(10).indices.tolist()
        bottom_logits = logits.topk(10, largest=False).indices.tolist()
    top_logits = [model.decode_token(i) for i in top_logits]
    bottom_logits = [model.decode_token(i) for i in bottom_logits]

    feature_vis = dict(
        index=index,
        examples_quantiles=examples_quantiles,
        bottom_logits=bottom_logits,
        top_logits=top_logits,
    )
    feature_path.write_text(json.dumps(feature_vis))
    bar.update(1)
    bar.refresh()

async for feature in ds:
    process_feature(feature)
bar.close()
# %%
