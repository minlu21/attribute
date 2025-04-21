import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.sparse_coder import SparseCoder

from .mlp_attribution import AttributionGraph

import fire


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotary_llama(x, position_ids, inv_freq, attention_scaling):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    with torch.autocast(device_type="cuda", enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def main(
    model_name="HuggingFaceTB/SmolLM2-135M",
    dataset="EleutherAI/fineweb-edu-dedup-10b",
    split="train",
    prompt="When John and Mary went to the store, John gave a bag to",
):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )

    # dataset = load_dataset(
    #                 dataset,
    #                 split=split)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = LanguageModel(model_name)
    # dataset = chunk_and_tokenize(
    #     dataset,
    #     tokenizer,
    #     max_seq_len=2048,
    #     num_proc=cpu_count() // 2,
    #     text_key="text",
    # )

    # data = dataset.select(range(0,1000))
    # dl = DataLoader(
    #     data,  # type: ignore
    #     batch_size=2, # Reduced batch size for potentially higher memory usage with hooks
    #     shuffle=False,
    # )

    transcoders = {}
    hookpoints_mlp = [
        f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)
    ]  # Use model config for layer count
    for hookpoint in hookpoints_mlp:
        temp_hookpoint = hookpoint.replace("model.layers.", "layers.")
        sae = SparseCoder.load_from_disk(
            f"/mnt/ssd-1/gpaulo/smollm-decomposition/sparsify/checkpoints/single_128x/{temp_hookpoint}",
            device="cuda",
        )
        transcoders[hookpoint] = sae
    # prompt = "Cow Cow Fish Fish Cat"

    attn_values = {}
    first_ln = {}
    pre_first_ln = {}
    second_ln = {}
    pre_second_ln = {}

    def get_attention_values_hook(module, input, output):
        # this hook is in the layer
        residual = input[0]
        pre_first_ln[module_to_name[module]] = residual
        layer_normed = module.input_layernorm(residual)
        first_ln[module_to_name[module]] = layer_normed

        # stuff related to attention
        input_shape = layer_normed.shape[:-1]
        hidden_shape = (*input_shape, -1, module.self_attn.head_dim)

        value_states = (
            module.self_attn.v_proj(layer_normed).view(hidden_shape).transpose(1, 2)
        )
        values = repeat_kv(value_states, 3)
        attn_values[module_to_name[module]] = values

    def get_ln2_hook(module, input, output):
        pre_second_ln[module_to_name[module]] = input[0]
        second_ln[module_to_name[module]] = output

    transcoder_activations = {}
    errors = {}
    skip_connections = {}
    input_norm = {}
    output_norm = {}

    def get_mlp_hook(module, input, output):

        if isinstance(input, tuple):
            input = input[0]

        module_name = module_to_name[module]
        transcoder = transcoders[module_name]
        # have to reshape input to lose the batch dimension
        input = input.view(-1, input.shape[-1])
        # have to normalize input
        input_norm[module_name] = input.norm(dim=1, keepdim=True)
        # input = input / input_norm[module_name]
        transcoder_acts = transcoder(input)
        # have to reshape output to get the batch dimension back
        transcoder_out = transcoder_acts.sae_out.view(output.shape)
        # have to unnormalize output
        transcoder_out_constant = 1  # output.norm(dim=2, keepdim=True)[0]
        output_norm[module_name] = transcoder_out_constant
        error = output - transcoder_out * output_norm[module_name]

        skip = input.to(transcoder.W_skip.dtype) @ transcoder.W_skip.mT
        activations = transcoder_acts.latent_acts * transcoder_out_constant
        transcoder_activations[module_name] = (
            activations,
            transcoder_acts.latent_indices,
        )

        skip_connections[module_name] = skip[0] * transcoder_out_constant
        errors[module_name] = error[0]

    # last_layer_activations = []
    # def get_last_layer_hook(module, input, output):

    #     last_layer_activations.append(output[0])

    hookpoints_layer = [
        f"model.layers.{i}" for i in range(model.config.num_hidden_layers)
    ]
    hookpoints_ln = [
        f"model.layers.{i}.post_attention_layernorm"
        for i in range(model.config.num_hidden_layers)
    ]
    name_to_module = {
        name: model.get_submodule(name)
        for name in hookpoints_layer + hookpoints_mlp + hookpoints_ln
    }
    module_to_name = {v: k for k, v in name_to_module.items()}
    for name, module in name_to_module.items():
        if ".mlp" in name:
            handle = module.register_forward_hook(get_mlp_hook)
        elif ".post_attention_layernorm" in name:
            handle = module.register_forward_hook(get_ln2_hook)
        else:
            handle = module.register_forward_hook(get_attention_values_hook)

    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model(
        **tokenized_prompt, output_attentions=True, output_hidden_states=True
    )
    attention_patterns = outputs.attentions
    logits = outputs.logits
    last_layer_activations = outputs.hidden_states[-1]
    last_layer_activations.retain_grad()

    # exit()
    attribution_graph = AttributionGraph(
        model,
        tokenizer,
        transcoders,
        tokenized_prompt["input_ids"][0],
        attention_patterns,
        first_ln,
        pre_first_ln,
        second_ln,
        pre_second_ln,
        input_norm,
        output_norm,
        attn_values,
        transcoder_activations,
        errors,
        skip_connections,
        logits,
        last_layer_activations,
        [transcoders[module].W_skip for module in hookpoints_mlp],
    )
    attribution_graph.initialize_graph()
    attribution_graph.flow()


fire.Fire(main)
