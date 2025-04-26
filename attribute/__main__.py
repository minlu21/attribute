from .mlp_attribution import AttributionGraph
from pathlib import Path
from .caching import TranscodedModel
import fire
from pathlib import Path
from .caching import TranscodedModel


async def main(
    prompt="When John and Mary went to the store, John gave a bag to",
    model_name="HuggingFaceTB/SmolLM2-135M",
    save_dir = Path("../attribution-graphs-frontend"),
    transcoder_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/sparsify/checkpoints/single_128x",
    cache_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/attribution_graph/results/transcoder_128x/latents",
):
    model_name = "HuggingFaceTB/SmolLM2-135M"
    hookpoint_fn = lambda hookpoint: hookpoint.replace("model.layers.", "layers.")
    model = TranscodedModel(
        model_name=model_name,
        transcoder_path=transcoder_path,
        hookpoint_fn=hookpoint_fn,
        device="cuda",
    )
    transcoded_outputs = model(prompt)
    attribution_graph = AttributionGraph(model, transcoded_outputs)
    attribution_graph.flow(2000)
    attribution_graph.save_graph(save_dir)
    await attribution_graph.cache_features(cache_path, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
