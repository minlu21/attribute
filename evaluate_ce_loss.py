import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.data import chunk_and_tokenize
from sparsify.sparse_coder import SparseCoder

model_name = "HuggingFaceTB/SmolLM2-135M"
dataset = "EleutherAI/fineweb-edu-dedup-10b"
split = "train"


model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        
    )

dataset = load_dataset(
                dataset,
                split=split)

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = chunk_and_tokenize(
    dataset,
    tokenizer,
    max_seq_len=2048,
    num_proc=cpu_count() // 2,
    text_key="text",
)

data = dataset.select(range(0,1000))
dl = DataLoader(
    data,  # type: ignore
    batch_size=16,
    shuffle=False,
)

hookpoints = [f"model.layers.{i}.mlp" for i in range(30)]

# First forward pass through the normal model, to compute the CE loss
# and collect the activations
input_dict = {}
output_dict = {}
name_to_module = {
            name: model.get_submodule(name) for name in hookpoints
}
module_to_name = {v: k for k, v in name_to_module.items()}

base_loss = 0
for batch in dl:
    with torch.no_grad():
        outputs = model(batch["input_ids"].to("cuda:0"),labels=batch["input_ids"])
        base_loss += outputs.loss.cpu().item()
base_loss /= len(dl)

per_layer_losses = {}
per_layer_fvu = {}
for hookpoint in tqdm(hookpoints):
    print(hookpoint)
    # Load the sparse coder
    try:
        sae = SparseCoder.load_from_disk(
            f"/mnt/ssd-1/gpaulo/smollm-decomposition/sparsify/checkpoints/single_128x/{hookpoint}",
            device="cuda:0",
        )
    except Exception as e:
        print(e)
        continue
    fvu = []
    def sae_hook(module, inputs, outputs):
        inputs = inputs[0]
        reshaped_inputs = inputs.flatten(0, 1)
        # normalize the inputs
        normed_inputs = reshaped_inputs / reshaped_inputs.norm(dim=1, keepdim=True)
        reshaped_outputs = outputs.flatten(0, 1)
        output_norm = reshaped_outputs.norm(dim=1, keepdim=True)
        print(output_norm.shape)
        normed_outputs = reshaped_outputs / output_norm
        output = sae.forward(normed_inputs,normed_outputs)
        print(output_norm)
        #print(output.fvu.cpu().mean().numpy())
        fvu.append(output.fvu.cpu().mean().numpy())
        new_outputs = output.sae_out
        #SAE out should be un-normalized, and put into the same shape as the output
        new_outputs = new_outputs * output_norm 
        new_outputs = new_outputs.view_as(outputs).to(outputs.dtype)
        return new_outputs

    handles = [
        name_to_module[hookpoint].register_forward_hook(sae_hook) 
    ]
    total_loss = 0
    for batch in dl:
        
        with torch.no_grad():
            outputs = model(batch["input_ids"].to("cuda:0"),labels=batch["input_ids"])
        loss = outputs.loss
        total_loss += loss.cpu().item()
       
    for handle in handles:
        handle.remove()
    per_layer_losses[hookpoint] = total_loss/len(dl)
    per_layer_fvu[hookpoint] = np.mean(fvu)

print("\nLoss Changes by Layer:")
print("-" * 40)
print(f"{'Layer':<20} {'Change %':>10} {'Fvu':>10}")
print("-" * 40)
for hookpoint, loss in per_layer_losses.items():
    pct_change = ((loss - base_loss) / base_loss) * 100
    print(f"{hookpoint[-20:]:20} {pct_change:>10.2f} {per_layer_fvu[hookpoint]:>10.2f}")
print("-" * 40)



