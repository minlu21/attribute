#%%
from IPython import get_ipython
if (ipy := get_ipython()) is not None:
    ipy.run_line_magic('load_ext', 'autoreload')
    ipy.run_line_magic('autoreload', '2')
    ipy.run_line_magic('env', 'CUDA_VISIBLE_DEVICES=1')
from attribute.mlp_attribution import AttributionGraph
from pathlib import Path
from attribute.caching import TranscodedModel
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
prompt = "The National Digital Analytics Group (N"
transcoded_outputs = model(prompt)
#%%
attribution_graph = AttributionGraph(model, transcoded_outputs)
attribution_graph.flow(2_000)
#%%
save_dir = Path("../attribution-graphs-frontend")
attribution_graph.save_graph(save_dir)
#%%
cache_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/attribution_graph/results/transcoder_128x/latents"
await attribution_graph.cache_features(cache_path, save_dir)
#%%
