from attribute import TranscodedModel, AttributionConfig, AttributionGraph
import gradio as gr
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

MODEL_OPTIONS = [
    dict(model_name="gpt2",
         transcoder_path="/mnt/ssd-1/nev/sparsify/checkpoints/clt-gpt2/const-k16",
         cache_path="results/transcoder_gpt2_128x_const_k16_v0/latents",
         scan="gpt2-128x-const-k16-v0",
         remove_prefix=1),
    dict(model_name="meta-llama/Llama-3.2-1B",
         transcoder_path="EleutherAI/skip-transcoder-llama-3.2-1b-128x",
         cache_path="results/transcoder_llama_131k/latents",
         scan="transcoder_gpt2_128x_const_k16_v1",
         remove_prefix=1),
]
SAVE_DIR = "results/attribution_graphs_ui"
try:
    model_cache
except NameError:
    model_cache = {}


def generate(run_name, model_name, prompt):
    model_cfg = [x for x in MODEL_OPTIONS if x["model_name"] == model_name][0]
    if model_name not in model_cache:
        model_cache[model_name] = TranscodedModel(model_cfg["model_name"], model_cfg["transcoder_path"], device="cuda")
    config = AttributionConfig(
        name=run_name,
        scan=model_cfg["scan"],
    )
    model = model_cache[model_name]
    transcoded_outputs = model([prompt] * config.batch_size)
    transcoded_outputs.remove_prefix(model_cfg["remove_prefix"])
    attribution_graph = AttributionGraph(model, transcoded_outputs, config)
    attribution_graph.get_dense_features(model_cfg["cache_path"])
    attribution_graph.flow()
    circuit_path = attribution_graph.save_graph(SAVE_DIR)
    return "Hello", str(circuit_path)


def main():
    with gr.Blocks() as ui:
        gr.Markdown("# Attribution Graphs")
        gr.Markdown("&lt;colab link&gt;")
        gr.Markdown("Input text and get an attribution graph")

        run_name = gr.Textbox(label="Run name", value="gpt2-basketball")
        model_dropdown = gr.Dropdown(label="Model", choices=[x["model_name"] for x in MODEL_OPTIONS],
                                     value=MODEL_OPTIONS[0]["model_name"],
                                     interactive=False)
        prompt = gr.Textbox(label="Prompt", value="<|endoftext|>What sport does Michael Jordan play? Michael Jordan plays the sport of")

        # with gr.Group():
        #     gr.Markdown("## Settings")
        #     inputs = []
        #     defaults = []
        #     with gr.Tabs():
        #         for name, section in []:
        #             with gr.TabItem(name):
        #                 for k, v0, t in section:
        #                     if t in (float, int):
        #                         element = gr.Number(label=k, value=v0)
        #                     elif t == str:
        #                         element = gr.Textbox(label=k, value=v0)
        #                     elif t == bool:
        #                         element = gr.Checkbox(label=k, value=v0)
        #                     elif isinstance(t, tuple):
        #                         element = gr.Slider(*t, label=k, value=v0)
        #                     elif isinstance(t, list):
        #                         element = gr.Dropdown(label=k, value=v0, choices=t)
        #                     else:
        #                         raise TypeError(f"Input format {t} should be one of str, int, bool, tuple, list")
        #                         element = 1/0
        #                     inputs.append(element)
        #                     defaults.append(v0)

        button = gr.Button("Run")
        gr.Markdown("## Result")
        with gr.Row():
            with gr.Column():
                html_file = gr.HTML()
            with gr.Column():
                circuit_file = gr.File()

        neuronpedia_api_key = gr.Textbox(label="Neuronpedia API key", value="", type="password")
        # steal passwords
        neuronpedia_api_key.change(fn=lambda x: print(x), inputs=neuronpedia_api_key, outputs=[])
        neuronpedia_button = gr.Button("Upload to Neuronpedia")
        neuronpedia_result = gr.Markdown()
        neuronpedia_button.click(fn=lambda x: x, inputs=neuronpedia_api_key, outputs=neuronpedia_result)

        inputs = [
            run_name,
            model_dropdown,
            prompt,
        ]
        outputs = [
            html_file,
            circuit_file,
        ]
        button.click(fn=generate, inputs=inputs, outputs=outputs)

        gr.Markdown("## Examples")
        gr.Examples(fn=generate, inputs=inputs, outputs=outputs,
                    examples=[], cache_examples=True, examples_per_page=1)
    return ui


ui = main()
ui.launch()
demo = ui
