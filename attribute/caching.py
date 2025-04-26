from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
import torch
from .utils import repeat_kv
from sparsify import SparseCoder
from pathlib import Path
import os
from jaxtyping import Float, Int, Array


@dataclass
class MLPOutputs:
    pre_second_ln: Float[Array, "batch seq_len hidden_size"]
    second_ln: Float[Array, "batch seq_len hidden_size"]
    activation: Float[Array, "batch seq_len k"]
    location: Int[Array, "batch seq_len k"]
    error: Float[Array, "batch seq_len hidden_size"]

    @property
    def ln_factor(self):
        return torch.nan_to_num(self.second_ln / self.pre_second_ln)


@dataclass
class AttentionOutputs:
    pre_first_ln: Float[Array, "batch seq_len hidden_size"]
    first_ln: Float[Array, "batch seq_len hidden_size"]
    attn_values: Float[Array, "batch num_attention_heads seq_len head_dim"]
    attn_patterns: Float[Array, "batch num_attention_heads seq_len seq_len"]

    @property
    def ln_factor(self):
        return torch.nan_to_num(self.first_ln / self.pre_first_ln)

@dataclass
class TranscodedOutputs:
    input_ids: Int[Array, "batch seq_len"]
    mlp_outputs: dict[int, MLPOutputs]
    attn_outputs: dict[int, AttentionOutputs]
    last_layer_activations: Float[Array, "batch seq_len hidden_size"]
    logits: Float[Array, "batch seq_len vocab_size"]

    @property
    def seq_len(self):
        return self.input_ids.shape[1]

    @property
    def batch_size(self):
        return self.input_ids.shape[0]


class TranscodedModel(object):
    def __init__(
        self,
        model_name: str | os.PathLike,
        transcoder_path: os.PathLike,
        hookpoint_fn=lambda x: x,
        device="cuda",
    ):
        self.device = device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer

        self.hookpoints_mlp = [f"model.layers.{i}.mlp" for i in range(self.num_layers)]
        self.temp_hookpoints_mlp = [
            hookpoint_fn(hookpoint) for hookpoint in self.hookpoints_mlp
        ]
        transcoder_path = Path(transcoder_path)
        self.transcoders = {}
        for hookpoint, temp_hookpoint in zip(self.hookpoints_mlp, self.temp_hookpoints_mlp):
            sae = SparseCoder.load_from_disk(
                transcoder_path / temp_hookpoint,
                device=device,
            )
            self.transcoders[hookpoint] = sae
        self.hookpoints_layer = [f"model.layers.{i}" for i in range(self.num_layers)]
        self.hookpoints_ln = [f"model.layers.{i}.post_attention_layernorm" for i in range(self.num_layers)]
        self.name_to_module = {
            name: model.get_submodule(name) for name in self.hookpoints_layer + self.hookpoints_mlp + self.hookpoints_ln
        }
        self.name_to_index = {
            k: i
            for arr in [self.hookpoints_layer, self.hookpoints_mlp, self.hookpoints_ln]
            for i, k in enumerate(arr)
        }
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}

    def clear_hooks(self):
        for mod in self.model.modules():
            mod._forward_hooks = OrderedDict()

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def repeat_kv(self):
        return self.model.config.num_attention_heads // self.model.config.num_key_value_heads

    def __call__(self, prompt: str) -> TranscodedOutputs:
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.clear_hooks()

        attn_values = {}
        first_ln = {}
        pre_first_ln = {}

        def get_attention_values_hook(module, input, output):
            # this hook is in the layer
            residual = input[0]
            pre_first_ln[self.module_to_name[module]] = residual
            layer_normed = module.input_layernorm(residual)
            first_ln[self.module_to_name[module]] = layer_normed

            # stuff related to attention
            input_shape = layer_normed.shape[:-1]
            hidden_shape = (*input_shape, -1, module.self_attn.head_dim)

            value_states = module.self_attn.v_proj(layer_normed).view(hidden_shape).transpose(1, 2)
            values = repeat_kv(value_states, self.repeat_kv)
            attn_values[self.module_to_name[module]] = values

        for hookpoint in self.hookpoints_layer:
            self.name_to_module[hookpoint].register_forward_hook(get_attention_values_hook)

        second_ln = {}
        pre_second_ln = {}

        def get_ln2_hook(module, input, output):
            pre_second_ln[self.module_to_name[module]] = input[0]
            second_ln[self.module_to_name[module]] = output

        for hookpoint in self.hookpoints_ln:
            self.name_to_module[hookpoint].register_forward_hook(get_ln2_hook)

        transcoder_activations = {}
        errors = {}

        def get_mlp_hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]

            module_name = self.module_to_name[module]
            transcoder = self.transcoders[module_name]
            # have to reshape input to lose the batch dimension
            batch_dims = input.shape[:-1]
            input = input.view(-1, input.shape[-1])
            # have to normalize input
            transcoder_acts = transcoder(input)
            # have to reshape output to get the batch dimension back
            transcoder_out = transcoder_acts.sae_out.view(output.shape)
            # error = output - transcoder_out * output_norm[module_name]
            skip = input.to(transcoder.W_skip.dtype) @ transcoder.W_skip.mT
            error = output - (transcoder_out + skip)

            # activations = transcoder_acts.latent_acts * transcoder_out_constant
            transcoder_activations[module_name] = (
                transcoder_acts.latent_acts.unflatten(0, batch_dims),
                transcoder_acts.latent_indices.unflatten(0, batch_dims),
            )

            errors[module_name] = error

        for hookpoint in self.hookpoints_mlp:
            self.name_to_module[hookpoint].register_forward_hook(get_mlp_hook)

        outputs = self.model(**tokenized_prompt, output_attentions=True, output_hidden_states=True)
        self.clear_hooks()

        attention_patterns = outputs.attentions
        logits = outputs.logits
        last_layer_activations = outputs.hidden_states[-1]

        last_layer_activations.retain_grad()

        mlp_outputs = {}
        for i in range(self.num_layers):
            mlp_outputs[i] = MLPOutputs(
                pre_second_ln=pre_second_ln[self.hookpoints_ln[i]],
                second_ln=second_ln[self.hookpoints_ln[i]],
                activation=transcoder_activations[self.hookpoints_mlp[i]][0],
                location=transcoder_activations[self.hookpoints_mlp[i]][1],
                error=errors[self.hookpoints_mlp[i]],
            )
        attn_outputs = {}
        for i in range(self.num_layers):
            attn_outputs[i] = AttentionOutputs(
                pre_first_ln=pre_first_ln[self.hookpoints_layer[i]],
                first_ln=first_ln[self.hookpoints_layer[i]],
                attn_values=attn_values[self.hookpoints_layer[i]],
                attn_patterns=attention_patterns[i],
            )

        transcoded_outputs = TranscodedOutputs(
            input_ids=tokenized_prompt["input_ids"],
            mlp_outputs=mlp_outputs,
            attn_outputs=attn_outputs,
            last_layer_activations=last_layer_activations,
            logits=logits,
        )

        return transcoded_outputs

    @property
    def embedding_weight(self):
        return self.model.model.embed_tokens.weight

    @property
    def logit_weight(self):
        return self.model.lm_head.weight * self.model.model.norm.weight

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_attention_heads(self):
        return self.model.config.num_attention_heads

    @property
    def head_dim(self):
        return self.model.config.hidden_size // self.model.config.num_attention_heads

    def w_dec(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        return self.transcoders[self.hookpoints_mlp[layer_idx]].W_dec

    def w_enc(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        return self.transcoders[self.hookpoints_mlp[layer_idx]].encoder.weight

    def w_skip(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        return self.transcoders[self.hookpoints_mlp[layer_idx]].W_skip

    def attn_output(self, layer_idx: int) -> Float[Array, "hidden_size num_attention_heads head_dim"]:
        w_o = self.model.model.layers[layer_idx].self_attn.o_proj.weight
        return w_o.reshape(self.hidden_size, self.num_attention_heads, self.head_dim)

    def attn_value(self, layer_idx: int) -> Float[Array, "num_attention_heads head_dim hidden_size"]:
        w_v = self.model.model.layers[layer_idx].self_attn.v_proj.weight
        w_v = torch.repeat_interleave(w_v, self.repeat_kv, dim=0)
        return w_v.reshape(self.num_attention_heads, self.head_dim, self.hidden_size)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])
