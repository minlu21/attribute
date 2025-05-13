import sys
from pathlib import Path

import fire
from loguru import logger

from .caching import TranscodedModel
from .mlp_attribution import AttributionConfig, AttributionGraph


async def main(
    prompt="When John and Mary went to the store, John gave a bag to",
    model_name="HuggingFaceTB/SmolLM2-135M",
    save_dir = Path("attribution-graphs-frontend"),
    transcoder_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/sparsify/checkpoints/single_128x",
    cache_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/attribution_graph/results/transcoder_128x/latents",
    name = "test-1-ts",
    scan = "default",
    remove_bos = False,
):
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = AttributionConfig(
        name=name,
        scan=scan,
    )
    model = TranscodedModel(
        model_name=model_name,
        transcoder_path=transcoder_path,
        device="cuda",
    )
    transcoded_outputs = model(prompt)

    if remove_bos:
        transcoded_outputs.input_ids = transcoded_outputs.input_ids[:, 1:]
        # only ever accessed w/ [-1], removing BOS doesn't matter
        # transcoded_outputs.last_layer_activations = transcoded_outputs.last_layer_activations[:, 1:]
        transcoded_outputs.logits = transcoded_outputs.logits[:, 1:]
        for k, mlp_output in transcoded_outputs.mlp_outputs.items():
            mlp_output.ln_factor = mlp_output.ln_factor[:, 1:]
            mlp_output.activation = mlp_output.activation[:, 1:]
            mlp_output.location = mlp_output.location[:, 1:]
            mlp_output.error = mlp_output.error[:, 1:]
        for k, attn_output in transcoded_outputs.attn_outputs.items():
            attn_output.ln_factor = attn_output.ln_factor[:, 1:]
            attn_output.attn_values = attn_output.attn_values[:, :, 1:]
            attn_output.attn_patterns = attn_output.attn_patterns[:, :, 1:, 1:]

    attribution_graph = AttributionGraph(model, transcoded_outputs, config)
    attribution_graph.flow()
    attribution_graph.save_graph(save_dir)
    await attribution_graph.cache_features(cache_path, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
