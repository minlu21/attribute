# %%
from IPython import get_ipython
from pathlib import Path

if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-2-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder
import sys
sys.path.append("..")
from attribute.caching import TranscodedModel

api = HfApi()
files = api.list_repo_files("google/gemma-scope-2b-pt-transcoders", repo_type="model")
output_path = Path("../results/gemma-scope-2b-pt-transcoders")
output_path.mkdir(parents=True, exist_ok=True)
for layer in tqdm(range(26)):
    prefix = f"layer_{layer}/width_16k/"
    filtered = [f for f in files if f.startswith(prefix)]
    filtered_l0 = [int(f.partition("average_l0_")[2].partition("/")[0]) for f in filtered]
    divergences = [abs(x - 100) for x in filtered_l0]
    min_divergence_index = np.argmin(divergences)
    min_divergence_file = filtered[min_divergence_index]
    print(layer, min_divergence_file)
    params_path = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        filename=min_divergence_file,
    )
    params_npz = np.load(params_path)
    W_enc = params_npz["W_enc"]
    W_dec = params_npz["W_dec"]
    b_dec = params_npz["b_dec"]
    b_enc = params_npz["b_enc"]
    threshold = params_npz["threshold"]
    b_enc_orig = b_enc
    b_enc, b_enc_post = b_enc - threshold, threshold

    W_enc = torch.from_numpy(W_enc.T).to(device)
    W_dec = torch.from_numpy(W_dec).to(device)
    b_dec = torch.from_numpy(b_dec).to(device)
    b_enc = torch.from_numpy(b_enc).to(device)
    b_enc_post = torch.from_numpy(b_enc_post).to(device)

    b_enc_orig = torch.from_numpy(b_enc_orig).to(device)
    threshold = b_enc_post

    d_in = b_dec.shape[0]
    d_f = b_enc.shape[0]

    pre_mlp_rmsnorm = 1.0 + model.model.layers[layer].pre_feedforward_layernorm.weight
    post_mlp_rmsnorm = 1.0 + model.model.layers[layer].post_feedforward_layernorm.weight

    W_enc = W_enc / pre_mlp_rmsnorm
    W_dec = W_dec / post_mlp_rmsnorm

    coder = SparseCoder(d_in, SparseCoderConfig(
        activation="topk",
        num_latents=d_f,
        k=128,
        transcode=True,
        skip_connection=False,
    ), device=device)
    coder.encoder.weight.data[:] = W_enc
    coder.encoder.bias.data[:] = b_enc
    coder.W_dec.data[:] = W_dec
    coder.b_dec.data[:] = b_dec
    coder.post_enc.data[:] = b_enc_post

    coder.save_to_disk(output_path / f"layers.{layer}.mlp")
#%%
model_hooked = TranscodedModel(
    model,
    output_path,
    device=device,
    post_ln_hook=True,
)

# %%
model_hooked("In another moment, down went Alice after it, never once considering how in the world she was to get out again.")
