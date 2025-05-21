import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from jaxtyping import Array, Float, Int
from sparsify import SparseCoder, CrossLayerRunner
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaPreTrainedModel
from transformers.models.gpt_neo import GPTNeoPreTrainedModel
from transformers.models.gpt2 import GPT2PreTrainedModel
from loguru import logger



GPT2Like = GPT2PreTrainedModel | GPTNeoPreTrainedModel


@dataclass
class MLPOutputs:
    ln_factor: Float[Array, "batch seq_len hidden_size"]
    activation: Float[Array, "batch seq_len k"]
    source_activation: Float[Array, "batch seq_len k"]
    location: Int[Array, "batch seq_len k"]
    error: Float[Array, "batch seq_len hidden_size"]
    source_error: Float[Array, "batch seq_len hidden_size"]
    l0: float


@dataclass
class TranscodedOutputs:
    input_ids: Int[Array, "batch seq_len"]
    mlp_outputs: dict[int, MLPOutputs]
    last_layer_activations: Float[Array, "batch seq_len hidden_size"]
    first_layer_activations: Float[Array, "batch seq_len hidden_size"]
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
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer

        if transcoder_path is None:
            return
        if hookpoint_fn is None:
            def hookpoint_fn(hookpoint):
                if isinstance(model, LlamaPreTrainedModel):
                    return hookpoint.replace("model.layers.", "layers.")
                elif isinstance(model, GPT2Like):
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
        self.hookpoints_attn_ln = [f"{self.layer_prefix}.{i}.{self.attn_layernorm_name}" for i in range(self.num_layers)]
        self.name_to_module = {
            name: model.get_submodule(name) for name in self.hookpoints_layer + self.hookpoints_mlp + self.hookpoints_ln + self.hookpoints_attn_ln
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

    def __call__(self, prompt: str | torch.Tensor, mask_features: dict[int, list[int]] = {},
                 errors_from: TranscodedOutputs | None = None,
                 no_error: bool = False) -> TranscodedOutputs:
        if isinstance(prompt, str):
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            logger.info(f"Tokenized prompt: {[self.decode_token(i) for i in tokenized_prompt.input_ids[0]]}")
        elif isinstance(prompt, torch.Tensor):
            tokenized_prompt = SimpleNamespace(input_ids=prompt.to(self.device))
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        self.clear_hooks()

        def ln_record_hook(module, input, output, save_dict=None):
            mean = input[0].mean(dim=-1, keepdim=True).detach()
            var = input[0].var(dim=-1, keepdim=True)
            multiplier = (1 / torch.sqrt(var + module.eps).detach()) * module.weight
            if save_dict is not None:
                save_dict[self.module_to_name[module]] = multiplier
            return (input[0] - mean) * multiplier + module.bias

        for hookpoint in self.hookpoints_layer:
            ln = getattr(self.name_to_module[hookpoint], self.attn_layernorm_name)
            ln.register_forward_hook(ln_record_hook)
            self.freeze_attention_pattern(hookpoint)

        second_ln = {}
        for hookpoint in self.hookpoints_ln:
            self.name_to_module[hookpoint].register_forward_hook(partial(ln_record_hook, save_dict=second_ln))

        runner = CrossLayerRunner()
        target_transcoder_activations = {}
        source_transcoder_activations = {}
        l0s = {}
        transcoder_locations = {}
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
            transcoder_acts = runner.encode(input, transcoder)

            l0 = (transcoder_acts.latent_acts != 0).float().sum(dim=-1).mean().item()
            l0s[module_name] = l0
            target_latent_acts = transcoder_acts.latent_acts.clone().unflatten(0, batch_dims)
            source_latent_acts = transcoder_acts.latent_acts
            source_latent_acts = source_latent_acts.unflatten(0, batch_dims).clone()
            source_latent_acts.detach_()
            source_latent_acts.requires_grad_(True)
            flat_source_acts = source_latent_acts.flatten(0, -2)
            transcoder_acts.latent_acts = flat_source_acts

            outputs = runner.decode(transcoder_acts, None, module_name)
            sae_out = outputs.sae_out

            layer_idx = self.name_to_index[module_name]
            masked_features = mask_features.get(layer_idx, [])
            if masked_features:
                acts = transcoder_acts.latent_acts
                indices = transcoder_acts.latent_indices
                # TODO: when patching, we can't use automatic attribution
                transcoder_acts.latent_acts = acts * (1 - torch.any(torch.stack([indices == i for i in masked_features], dim=0), dim=0).float())

            # have to reshape output to get the batch dimension back
            transcoder_out = sae_out.view(output.shape)
            diff = output - transcoder_out

            if no_error:
                error = torch.zeros_like(output)
            elif errors_from is None:
                error = diff
            else:
                error = errors_from.mlp_outputs[layer_idx].error
            error = error.clone()
            error.detach_()
            error.requires_grad_(True)
            logger.info(f"Layer {module_name} error: {diff.norm() / output.norm()}")

            latent_indices = transcoder_acts.latent_indices.unflatten(0, batch_dims)
            target_transcoder_activations[module_name] = target_latent_acts
            source_transcoder_activations[module_name] = source_latent_acts
            transcoder_locations[module_name] = latent_indices

            result = (transcoder_out + error).to(output)
            errors[module_name] = error
            return result

        for hookpoint in self.hookpoints_mlp:
            self.name_to_module[hookpoint].register_forward_hook(get_mlp_hook)

        outputs = self.model(input_ids=tokenized_prompt.input_ids,
                            #  attention_mask=tokenized_prompt.attention_mask,
                             output_hidden_states=True)
        self.clear_hooks()

        logits = outputs.logits

        logger.info("Top last token logits:")
        for index in logits[0, -1].topk(10).indices:
            logger.info(f"{self.decode_token(index)}: {logits[0, -1][index].item()}")

        first_layer_activations = outputs.hidden_states[0]
        if first_layer_activations.requires_grad:
            first_layer_activations.retain_grad()

        last_layer_activations = outputs.hidden_states[-1]
        if last_layer_activations.requires_grad:
            last_layer_activations.retain_grad()

        runner.reset()

        mlp_outputs = {}
        for i in range(self.num_layers):
            mlp_outputs[i] = MLPOutputs(
                ln_factor=second_ln[self.hookpoints_ln[i]],
                activation=target_transcoder_activations[self.hookpoints_mlp[i]],
                source_activation=source_transcoder_activations[self.hookpoints_mlp[i]],
                location=transcoder_locations[self.hookpoints_mlp[i]],
                error=errors[self.hookpoints_mlp[i]],
                source_error=errors[self.hookpoints_mlp[i]],
                l0=l0s[self.hookpoints_mlp[i]],
            )

        transcoded_outputs = TranscodedOutputs(
            input_ids=tokenized_prompt.input_ids,
            mlp_outputs=mlp_outputs,
            first_layer_activations=first_layer_activations,
            last_layer_activations=last_layer_activations,
            logits=logits,
        )

        return transcoded_outputs

    @property
    def layer_prefix(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "model.layers"
        elif isinstance(self.model, GPT2Like):
            return "transformer.h"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def attn_layernorm_name(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "input_layernorm"
        elif isinstance(self.model, GPT2Like):
            return "ln_1"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_layernorm_name(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "post_attention_layernorm"
        elif isinstance(self.model, GPT2Like):
            return "ln_2"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_in_proj_name(self):
        if isinstance(self.model, GPT2Like):
            return "c_proj"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def embedding_weight(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.model.embed_tokens.weight
        elif isinstance(self.model, GPT2Like):
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
        elif isinstance(self.model, GPT2Like):
            return self.model.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_weight(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.lm_head.weight * self.final_ln.weight
        elif isinstance(self.model, GPT2Like):
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
            if target_layer_idx != layer_idx:
                raise AttributeError
            return self.transcoders[self.hookpoints_mlp[layer_idx]].W_dec
        except AttributeError:
            if target_layer_idx is None:
                logger.warning("Summing decoder weights because target_layer_idx is None")
                target_layer_idx = layer_idx
                weight_combined = torch.zeros((self.w_dec(layer_idx, layer_idx).shape[0], self.hidden_size,), device=self.device, dtype=torch.float32)
                for target_layer_idx in range(layer_idx, self.num_layers):
                    try:
                        weight_combined += self.w_dec(layer_idx, target_layer_idx)
                    except IndexError:
                        break
                # weights_at_layers = {}
                # while target_layer_idx < self.num_layers:
                #     weights_at_layers[target_layer_idx] = weight_combined
                #     for layer_from, weight_at in weights_at_layers.items():
                #         try:
                #             weight_combined += weight_at @ self.w_skip(layer_from, target_layer_idx).T
                #         except IndexError:
                #             pass
                #     try:
                #         weight_combined += self.w_dec(layer_idx, target_layer_idx)
                #     except IndexError:
                #         pass
                #     target_layer_idx += 1
                return weight_combined
            # assume the target layer is contiguous
            assert target_layer_idx >= layer_idx
            try:
                return self.transcoders[self.hookpoints_mlp[layer_idx]].W_decs[target_layer_idx - layer_idx]
            except AttributeError:
                raise IndexError

    def w_skip(self, layer_idx: int, target_layer_idx: int | None = None) -> Float[Array, "hidden_size hidden_size"]:
        transcoder = self.transcoders[self.hookpoints_mlp[layer_idx]]
        try:
            transcoder.W_skip
        except AttributeError:
            assert target_layer_idx is not None, "target_layer_idx must be provided for multi-target transcoders"
            assert target_layer_idx >= layer_idx
            try:
                w_skip = transcoder.W_skips[target_layer_idx - layer_idx]
            except AttributeError:
                raise IndexError
            if w_skip is None:
                raise IndexError
            return w_skip
        else:
            if target_layer_idx != layer_idx:
                raise IndexError
            return transcoder.W_skip

    def w_enc(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        return self.transcoders[self.hookpoints_mlp[layer_idx]].encoder.weight

    def attn(self, layer_idx: int) -> torch.nn.Module:
        layer = self.model.get_submodule(self.layer_prefix)[layer_idx]
        if isinstance(self.model, LlamaPreTrainedModel):
            return layer.self_attn
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return layer.attn.attention
        elif isinstance(self.model, GPT2PreTrainedModel):
            return layer.attn
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def attn_output(self, layer_idx: int) -> Float[Array, "hidden_size num_attention_heads head_dim"]:
        if isinstance(self.model, LlamaPreTrainedModel):
            w_o = self.attn(layer_idx).o_proj.weight
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            w_o = self.attn(layer_idx).out_proj.weight
        elif isinstance(self.model, GPT2PreTrainedModel):
            w_o = self.attn(layer_idx).c_proj.weight.T
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        return w_o.reshape(self.hidden_size, self.num_attention_heads, self.head_dim)

    def attn_value(self, layer_idx: int) -> Float[Array, "num_attention_heads head_dim hidden_size"]:
        if not isinstance(self.model, GPT2PreTrainedModel):
            w_v = self.attn(layer_idx).v_proj.weight
        else:
            w_q, w_k, w_v = torch.split(self.attn(layer_idx).c_attn.weight, self.hidden_size, dim=1)
            w_v = w_v.T
        w_v = torch.repeat_interleave(w_v, self.repeat_kv, dim=0)
        return w_v.reshape(self.num_attention_heads, self.head_dim, self.hidden_size)

    def project_v(self, layer_idx: int, layer_normed: Float[Array, "batch seq_len hidden_size"]) -> Float[Array, "batch seq_len num_attention_heads head_dim"]:
        if not isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).v_proj(layer_normed)
        else:
            projected = self.attn(layer_idx).c_attn(layer_normed)
            q, k, v = torch.split(projected, self.hidden_size, dim=-1)
            return v

    def attn_q_slice(self, layer_idx: int):
        if isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).c_attn, 0, self.hidden_size
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return self.attn(layer_idx).q_proj, 0, self.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def attn_k_slice(self, layer_idx: int):
        if isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).c_attn, self.hidden_size, self.hidden_size * 2
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return self.attn(layer_idx).k_proj, 0, self.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def freeze_attention_pattern(self, hookpoint: str):
        index = self.name_to_index[hookpoint]
        def freeze_slice(module, input, output, start, end):
            if start == 0 and end == output.shape[-1]:
                return output.detach()
            indices = torch.arange(output.shape[-1], device=output.device)
            mask = (indices >= start) & (indices < end)
            output = torch.where(mask, output.detach(), output)
            return output
        for module, start, end in (self.attn_q_slice(index), self.attn_k_slice(index)):
            module.register_forward_hook(partial(freeze_slice,
                                                 start=torch.tensor(start, device=self.device),
                                                 end=torch.tensor(end, device=self.device)))
