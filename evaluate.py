#%%
import IPython
if ip := IPython.get_ipython():
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

from attribute.caching import TranscodedModel
from sparsify.data import chunk_and_tokenize
from datasets import load_dataset
from multiprocessing import cpu_count
from tqdm.auto import tqdm
import torch
from loguru import logger

model = TranscodedModel(
    "nev/GELU_4L512W_C4_Code",
    transcoder_path="../e2e/checkpoints/gelu-4l-clt/ef128k64",
)
#%%
dataset = load_dataset("NeelNanda/pile-10k", split="train")
dataset = chunk_and_tokenize(
    dataset,
    model.tokenizer,
    max_seq_len=128,
    num_proc=cpu_count() // 2,
    text_key="text",
    return_overflowed_tokens=True,
)
# %%
logger.remove()
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=cpu_count() // 2)
for batch in (bar := tqdm(dataloader)):
    input_ids = batch["input_ids"].to(model.device)
    out = model(input_ids[:, :-1], no_error=True)
    logits = out.logits
    logits = logits.argmax(dim=-1)
    accuracy = (logits == input_ids[:, 1:]).float().mean().item()
    print(f"Accuracy: {accuracy}")
    break
# %%
