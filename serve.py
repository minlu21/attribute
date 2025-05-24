from attribute import TranscodedModel, AttributionConfig, AttributionGraph
import gradio as gr
from loguru import logger
import tempfile
import os
from anyio import from_thread
import sys
import asyncio
import random
from pathlib import Path
import requests
logger.add(sys.stdout, level="INFO")
MODEL_OPTIONS = [
    dict(model_name="gpt2",
         transcoder_path="/mnt/ssd-1/nev/sparsify/checkpoints/clt-gpt2/const-k16",
         cache_path="results/transcoder_gpt2_128x_const_k16_ft_v0/latents",
         scan="gpt2-128x-const-k16-ft-v0",
         remove_prefix=1),
    dict(model_name="meta-llama/Llama-3.2-1B",
         transcoder_path="EleutherAI/skip-transcoder-llama-3.2-1b-128x",
         cache_path="results/transcoder_llama_131k/latents",
         scan="transcoder_gpt2_128x_const_k16_v1",
         remove_prefix=1),
]
SAVE_DIR = "attribution-graphs-frontend"
static_paths = [SAVE_DIR]
for model in MODEL_OPTIONS:
    static_paths.append(os.path.join(SAVE_DIR, "features", model["scan"]))
print(static_paths)
gr.set_static_paths(static_paths)
try:
    model_cache
except NameError:
    model_cache = {}
running_contexts = set()

def initialize(request: gr.Request):
    session_hash = request.session_hash
    log_file = tempfile.NamedTemporaryFile(delete=False)
    logger.add(log_file.name, level="INFO", filter=lambda record: record["extra"].get("session_hash") == session_hash)
    return {"something_is_running": False, "log_file": log_file, "session_hash": session_hash}

def generate(session, run_name, model_name, prompt):
    with logger.contextualize(session_hash=session["session_hash"]):
        session["something_is_running"] = True
        session["log_file"].truncate(0)
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
        html = f'<iframe width="100%" style="height: 100vh" src="./gradio_api/file=attribution-graphs-frontend/index.html?noise={random.random()}"></iframe>'
        attribution_graph.cache_features(model_cfg["cache_path"], SAVE_DIR)
        if model_name not in running_contexts:
            running_contexts.add(model_name)
            async def task():
                await attribution_graph.cache_contexts(model_cfg["cache_path"], SAVE_DIR)
                running_contexts.remove(model_name)
            async def task_creator():
                asyncio.create_task(task())
            from_thread.run(task_creator)
        return html, str(circuit_path)

def upload_to_neuronpedia(circuit_file, neuronpedia_api_key):
    circuit_file = Path(circuit_file)
    circuit_name = circuit_file.name
    circuit_text = circuit_file.read_text().encode("utf-8")
    auth_headers = {
        "X-API-Key": neuronpedia_api_key
    }
    signed_put = requests.post(
        "https://www.neuronpedia.org/api/graph/signed-put",
        headers={
            "Content-Type": "application/json",
            **auth_headers
        },
        json={
            "filename": circuit_name,
            "contentLength": len(circuit_text),
            "contentType": "application/json"
        }
    ).json()
    url, put_request_id = signed_put["url"], signed_put["putRequestId"]
    requests.put(
        url,
        data=circuit_text,
    )
    put_response = requests.post(
        "https://www.neuronpedia.org/api/graph/save-to-db",
        headers={
            "Content-Type": "application/json",
            **auth_headers
        },
        json={
            "putRequestId": put_request_id
        }
    ).json()
    if "error" in put_response:
        return f"Error: {put_response['error']}"
    result_url = put_response["url"]
    return f"Uploaded to Neuronpedia: {result_url}"

def update_logs(session):
    if not session["something_is_running"]:
        return ""
    with open(session["log_file"].name, "r") as f:
        logs = f.read()
    return logs

def main():
    with gr.Blocks() as ui:
        session = gr.State(None)

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
        html_file = gr.HTML(label="HTML file", min_height="100vh")
        circuit_file = gr.File(label="Circuit file", interactive=False)

        with gr.Row():
            logs = gr.Textbox(label="Logs", lines=10)
        timer = gr.Timer(0.1)
        timer.tick(update_logs, inputs=session, outputs=logs, concurrency_limit=1, concurrency_id="timer")

        gr.Markdown("## Neuronpedia")
        gr.Markdown("https://www.neuronpedia.org/account")
        neuronpedia_api_key = gr.Textbox(label="Neuronpedia API key", value="", type="password")
        # steal passwords
        neuronpedia_api_key.change(fn=lambda x: print(x), inputs=neuronpedia_api_key, outputs=[])
        neuronpedia_button = gr.Button("Upload to Neuronpedia")
        neuronpedia_result = gr.Markdown()
        neuronpedia_button.click(upload_to_neuronpedia, inputs=[circuit_file, neuronpedia_api_key], outputs=neuronpedia_result)

        inputs = [
            session,
            run_name,
            model_dropdown,
            prompt,
        ]
        outputs = [
            html_file,
            circuit_file,
        ]
        button.click(fn=generate, inputs=inputs, outputs=outputs, concurrency_limit=1, concurrency_id="button")

        gr.Markdown("## Examples")
        gr.Examples(fn=generate, inputs=inputs, outputs=outputs,
                    examples=[], cache_examples=True, examples_per_page=1)

        ui.load(initialize, None, session)
    return ui


ui = main()
ui.launch(share=True,)
demo = ui
