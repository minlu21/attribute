import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from jaxtyping import Array, Float, Int
from sparsify import SparseCoder
from sparsify.utils import decoder_impl
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaPreTrainedModel
from transformers.models.gpt_neo import GPTNeoPreTrainedModel
from loguru import logger

from .utils import repeat_kv


@dataclass
class MLPOutputs:
    pre_second_ln: Float[Array, "batch seq_len hidden_size"]
    second_ln: Float[Array, "batch seq_len hidden_size"]
    activation: Float[Array, "batch seq_len k"]
    location: Int[Array, "batch seq_len k"]
    error: Float[Array, "batch seq_len hidden_size"]

    @property
    def ln_factor(self):
        return torch.nan_to_num(self.second_ln / self.pre_second_ln, nan=1.0, posinf=1.0, neginf=1.0)


@dataclass
class AttentionOutputs:
    pre_first_ln: Float[Array, "batch seq_len hidden_size"]
    first_ln: Float[Array, "batch seq_len hidden_size"]
    attn_values: Float[Array, "batch num_attention_heads seq_len head_dim"]
    attn_patterns: Float[Array, "batch num_attention_heads seq_len seq_len"]

    @property
    def ln_factor(self):
        return torch.nan_to_num(self.first_ln / self.pre_first_ln, nan=1.0, posinf=1.0, neginf=1.0)

@dataclass
class TranscodedOutputs:
    input_ids: Int[Array, "batch seq_len"]
    mlp_outputs: dict[int, MLPOutputs]
    attn_outputs: dict[int, AttentionOutputs]
    last_layer_activations: Float[Array, "batch seq_len hidden_size"]
    pre_final_ln: Float[Array, "batch seq_len hidden_size"]
    final_ln: Float[Array, "batch seq_len hidden_size"]
    logits: Float[Array, "batch seq_len vocab_size"]

    @property
    def final_ln_factor(self):
        return torch.nan_to_num(self.final_ln / self.pre_final_ln, nan=1.0, posinf=1.0, neginf=1.0)

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
        hookpoint_fn=None,
        device="cuda",
    ):
        logger.info(f"Loading model {model_name} on device {device}")
        self.device = device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer

        if hookpoint_fn is None:
            def hookpoint_fn(hookpoint):
                if isinstance(model, LlamaPreTrainedModel):
                    return hookpoint.replace("model.layers.", "layers.")
                elif isinstance(model, GPTNeoPreTrainedModel):
                    return hookpoint.replace("transformer.h.", "h.")
                else:
                    logger.warning(f"Unknown model type: {type(model)}. Using default hookpoint.")
                    return hookpoint
        self.hookpoints_mlp = [f"{self.layer_prefix}.{i}.mlp" for i in range(self.num_layers)]
        self.temp_hookpoints_mlp = [
            hookpoint_fn(hookpoint) for hookpoint in self.hookpoints_mlp
        ]
        logger.info(f"Loading transcoders from {transcoder_path}")
        transcoder_path = Path(transcoder_path)
        self.transcoders = {}
        for hookpoint, temp_hookpoint in zip(self.hookpoints_mlp, self.temp_hookpoints_mlp):
            sae = SparseCoder.load_from_disk(
                transcoder_path / temp_hookpoint,
                device=device,
            )
            self.transcoders[hookpoint] = sae
        self.hookpoints_layer = [f"{self.layer_prefix}.{i}" for i in range(self.num_layers)]
        self.hookpoints_ln = [f"{self.layer_prefix}.{i}.{self.mlp_layernorm_name}" for i in range(self.num_layers)]
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
        if not hasattr(self.model.config, "num_key_value_heads"):
            return 1
        return self.model.config.num_attention_heads // self.model.config.num_key_value_heads

    def __call__(self, prompt: str, mask_features: dict[int, list[int]] = {}) -> TranscodedOutputs:
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.clear_hooks()

        attn_values = {}
        first_ln = {}
        pre_first_ln = {}

        def get_attention_values_hook(module, input, output):
            # this hook is in the layer
            residual = input[0]
            layer_name = self.module_to_name[module]
            pre_first_ln[layer_name] = residual
            layer_normed = getattr(module, self.attn_layernorm_name)(residual)
            first_ln[layer_name] = layer_normed

            # stuff related to attention
            input_shape = layer_normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            layer_idx = self.name_to_index[layer_name]
            value_states = self.attn(layer_idx).v_proj(layer_normed).view(hidden_shape).transpose(1, 2)
            values = repeat_kv(value_states, self.repeat_kv)
            attn_values[layer_name] = values

        for hookpoint in self.hookpoints_layer:
            self.name_to_module[hookpoint].register_forward_hook(get_attention_values_hook)

        second_ln = {}
        pre_second_ln = {}

        def get_ln2_hook(module, input, output):
            pre_second_ln[self.module_to_name[module]] = input[0]
            second_ln[self.module_to_name[module]] = output

        for hookpoint in self.hookpoints_ln:
            self.name_to_module[hookpoint].register_forward_hook(get_ln2_hook)

        transcoder_outputs = {}
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
            transcoder_outputs[module_name] = transcoder_acts

            sae_out = 0
            to_delete = set()
            subtract = output.new_zeros(np.prod(output.shape[:-1]), output.shape[-1])
            for k, v in transcoder_outputs.items():
                layer_idx = self.name_to_index[k]
                masked_features = mask_features.get(layer_idx, [])
                if masked_features:
                    acts = v.latent_acts
                    indices = v.latent_indices
                    acts = acts * torch.any(torch.stack([indices == i for i in masked_features], dim=0), dim=0).float()
                    decoded = decoder_impl(
                        acts, indices, v.current_w_dec
                    )
                    subtract += decoded

                out = v(
                    None,
                    addition=(0 if k != module_name else sae_out) / max(1, len(transcoder_outputs) - 1),
                )
                if k == module_name:
                    sae_out = out.sae_out
                else:
                    sae_out += out.sae_out
                if out.is_last:
                    to_delete.add(k)
            for k in to_delete:
                del transcoder_outputs[k]
            # have to reshape output to get the batch dimension back
            transcoder_out = sae_out.view(output.shape)
            error = output - transcoder_out
            logger.info(f"Layer {module_name} error: {error.norm() / output.norm()}")

            transcoder_activations[module_name] = (
                transcoder_acts.latent_acts.unflatten(0, batch_dims),
                transcoder_acts.latent_indices.unflatten(0, batch_dims),
            )

            errors[module_name] = error

            return output - subtract.unflatten(0, batch_dims)

        for hookpoint in self.hookpoints_mlp:
            self.name_to_module[hookpoint].register_forward_hook(get_mlp_hook)

        pre_final_ln, final_ln = None, None
        def get_final_ln_hook(module, input, output):
            nonlocal pre_final_ln, final_ln
            pre_final_ln = input[0]
            final_ln = output
        self.final_ln.register_forward_hook(get_final_ln_hook)

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
            pre_final_ln=pre_final_ln,
            final_ln=final_ln,
            logits=logits,
        )

        return transcoded_outputs

    @property
    def layer_prefix(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "model.layers"
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return "transformer.h"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def attn_layernorm_name(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "input_layernorm"
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return "ln_1"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_layernorm_name(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "post_attention_layernorm"
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return "ln_2"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def embedding_weight(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.model.embed_tokens.weight
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return self.model.transformer.wte.weight
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def parallel_attn(self):
        return isinstance(self.model, GPTNeoPreTrainedModel)

    @property
    def final_ln(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.model.norm
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return self.model.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_weight(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.lm_head.weight * self.final_ln.weight
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return self.model.lm_head.weight * self.final_ln.weight
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_bias(self):
        bias = self.model.lm_head.bias
        if bias is None:
            bias = 0
        return bias + self.logit_weight @ self.final_ln.bias

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_attention_heads(self):
        return self.model.config.num_attention_heads

    @property
    def head_dim(self):
        return self.model.config.hidden_size // self.model.config.num_attention_heads

    @torch.no_grad()
    @torch.autocast("cuda")
    def w_dec(self, layer_idx: int, target_layer_idx: int | None = None) -> Float[Array, "features hidden_size"]:
        try:
            return self.transcoders[self.hookpoints_mlp[layer_idx]].W_dec
        except AttributeError:
            if target_layer_idx is None:
                logger.warning("Summing decoder weights because target_layer_idx is None")
                target_layer_idx = layer_idx
                weight_combined = torch.zeros((self.w_dec(layer_idx, layer_idx).shape[0], self.hidden_size,), device=self.device, dtype=torch.float32)
                weights_at_layers = {}
                while target_layer_idx < self.num_layers:
                    weights_at_layers[target_layer_idx] = weight_combined
                    for layer_from, weight_at in weights_at_layers.items():
                        weight_combined += weight_at @ self.w_skip(layer_from, target_layer_idx).T
                    weight_combined += self.w_dec(layer_idx, target_layer_idx)
                    target_layer_idx += 1
                return weight_combined
            # assume the target layer is contiguous
            assert target_layer_idx >= layer_idx
            return self.transcoders[self.hookpoints_mlp[layer_idx]].W_decs[target_layer_idx - layer_idx]

    def w_skip(self, layer_idx: int, target_layer_idx: int | None = None) -> Float[Array, "hidden_size hidden_size"]:
        try:
            self.hookpoints_mlp[layer_idx].W_skip
        except AttributeError:
            assert target_layer_idx is not None, "target_layer_idx must be provided for multi-target transcoders"
            assert target_layer_idx >= layer_idx
            w_skip = self.transcoders[self.hookpoints_mlp[layer_idx]].W_skips[target_layer_idx - layer_idx]
            return w_skip
        else:
            if target_layer_idx != layer_idx:
                raise IndexError
            return self.transcoders[self.hookpoints_mlp[layer_idx]].W_skip

    def w_enc(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        return self.transcoders[self.hookpoints_mlp[layer_idx]].encoder.weight

    def attn(self, layer_idx: int) -> torch.nn.Module:
        layer = self.model.get_submodule(self.layer_prefix)[layer_idx]
        if isinstance(self.model, LlamaPreTrainedModel):
            return layer.self_attn
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return layer.attn.attention
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def attn_output(self, layer_idx: int) -> Float[Array, "hidden_size num_attention_heads head_dim"]:
        if isinstance(self.model, LlamaPreTrainedModel):
            w_o = self.attn(layer_idx).o_proj.weight
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            w_o = self.attn(layer_idx).out_proj.weight
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        return w_o.reshape(self.hidden_size, self.num_attention_heads, self.head_dim)

    def attn_value(self, layer_idx: int) -> Float[Array, "num_attention_heads head_dim hidden_size"]:
        w_v = self.attn(layer_idx).v_proj.weight
        w_v = torch.repeat_interleave(w_v, self.repeat_kv, dim=0)
        return w_v.reshape(self.num_attention_heads, self.head_dim, self.hidden_size)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])
