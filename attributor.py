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
edges = attribution_graph.edges
edge_keys = [edge.partition(" -> ")[::2] for edge in edges]
#%%
nodes = []
indices = {}
for i, node in enumerate(set(x for pair in edge_keys for x in pair)):
    nodes.append(node)
    indices[node] = i
# %%
matrix = np.zeros((len(nodes), len(nodes)))
for i, edge in enumerate(tqdm(edges)):
    source, target = edge_keys[i]
    source_node = source.split(".")[-1]
    matrix[indices[target], indices[source]] = abs(edges[edge].weight)
matrix /= np.maximum(1e-2, matrix.sum(axis=1, keepdims=True))
#%%
def cg(x, cf=500):
    x = x[:x.shape[0] // cf * cf, :x.shape[1] // cf * cf]
    x = x.reshape(x.shape[0] // cf, cf, x.shape[1] // cf, cf)
    # x = x.max(axis=1).max(axis=-1)
    # x = x.mean(axis=1).max(axis=-1)
    x = x.mean(axis=1).mean(axis=-1)
    return x
coarse = cg(matrix)
plt.imshow(np.log10(coarse))
# plt.imshow(np.log10(coarse) >= -2.0)
plt.colorbar()
plt.show()
#%%
sources = matrix.sum(axis=0)
plt.plot(np.sort(sources))
plt.yscale("log")
#%%
I = np.eye(len(nodes))
%time influence = np.linalg.inv(I - matrix) - I
#%%
plt.imshow(np.log10(cg(np.abs(influence - matrix))))
plt.colorbar()
plt.show()
# %%
# compute logit influence
influence_sources = np.zeros((len(nodes),))
for index, node in enumerate(nodes):
    if "output" not in node:
        continue
    influence_sources[index] = attribution_graph.nodes[node].probability
#%%
# usage = influence_sources @ matrix
usage = influence_sources @ influence
plt.plot(np.sort(usage))
plt.yscale("log")
#%%
for i in usage.argsort()[-10:]:
    print(nodes[i], usage[i])
#%%
# node_thresold = 1e-3
# node_thresold = 5e-3
# node_thresold = 5e-3
node_thresold = 1e-2
secondary_threshold = 1e-4
per_layer_position = 2
edge_threshold = 1e-3
#%%
# select used node names
selected_nodes = [node for i, node in enumerate(nodes)
                  if attribution_graph.nodes[node].node_type in ("InputNode", "OutputNode", "ErrorNode")]
for seq_idx in range(transcoded_outputs.input_ids.shape[-1]):
    for layer_idx in range(model.num_layers):
        matching_nodes = [
            node for node in nodes
            if attribution_graph.nodes[node].layer_index == layer_idx
            and attribution_graph.nodes[node].token_position == seq_idx
            and node not in selected_nodes
        ]
        matching_nodes.sort(key=lambda x: usage[indices[x]], reverse=True)
        matching_nodes = matching_nodes[:per_layer_position] + [
            node
            for node in matching_nodes[per_layer_position:]
            if usage[indices[node]] > node_thresold
        ]
        matching_nodes = [
            node for node in matching_nodes
            if usage[indices[node]] > secondary_threshold
        ]
        selected_nodes.extend(matching_nodes)
# selected_nodes = [node for i, node in enumerate(nodes)
#                   if usage[i] > node_thresold
#                   or attribution_graph.nodes[node].node_type in ("InputNode", "OutputNode")
#                   ]
export_nodes = []
for n in selected_nodes:
    export_nodes.append(attribution_graph.nodes[n])
#%%
export_edges = []
for eid, e in attribution_graph.edges.items():
    if not (e.source.id in selected_nodes and e.target.id in selected_nodes):
        continue
    if abs(influence[indices[e.target.id], indices[e.source.id]]) < edge_threshold:
        continue
    export_edges.append(e)
#%%
len(export_nodes), len(export_edges)
#%%

save_dir = Path("../attribution-graphs-frontend")

tokens = [model.decode_token(i) for i in transcoded_outputs.input_ids[0].tolist()]
name = "test-1"
# prefix = ["<EOT>"]
prefix = []
scan = "default"
metadata = dict(
    slug=name,
    # scan="jackl-circuits-runs-1-4-sofa-v3_0",
    scan=scan,
    prompt_tokens=prefix + tokens,
    prompt="".join(tokens),
    title_prefix="",
    n_layers=model.num_layers,
)

nodes_json = [
    dict(
        feature_type=dict(
            IntermediateNode="cross layer transcoder",
            OutputNode="logit",
            InputNode="embedding",
            ErrorNode="mlp reconstruction error",
        )[node.node_type],
        layer=(str(node.layer_index)
        if node.layer_index != model.num_layers else "-1")
        if node.layer_index != -1 else "E",
        node_id=node.id_js,
        jsNodeId=node.id_js + "-0",
        feature=int(node.feature_index) if hasattr(node, "feature_index") else None,
        ctx_idx=node.token_position + len(prefix),
        run_idx=0, reverse_ctx_idx=0,
        clerp="" if not isinstance(node, OutputNode) else f"output: \"{node.token_str}\" (p={node.probability:.3f})",
        token_prob=0.0 if not isinstance(node, OutputNode) else node.probability,
        is_target_logit=False,
    ) for node in export_nodes
]


for node in nodes_json:
    if node["feature_type"] != "cross layer transcoder":
        continue
    l, f = int(node["layer"]), int(node["feature"])
    idx_cantor = cantor(l, f)
    node["feature"] = idx_cantor

links_json = [
    e.to_dict() for e in export_edges
]

result = dict(
    metadata=metadata,
    nodes=nodes_json,
    links=links_json,
    qParams=dict(
        linkType="both",
        # # node IDs
        # pinnedIds=[],
        # # clickedId=export_nodes[0].id,
        # clickedId=None,
        # supernodes=[],
        # # x/y position pairs for supernodes, separated by spaces
        # sg_pos="",
    )
)

open(save_dir / "graph_data" / f"{name}.json", "w").write(json.dumps(result))
open(save_dir / "data/graph-metadata.json", "w").write(json.dumps(dict(graphs=[metadata])))
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
