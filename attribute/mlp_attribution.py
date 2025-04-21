from dataclasses import dataclass, replace
import torch
from typing import Optional
from heapq import heappush, heappop
from functools import lru_cache
from torch import Tensor
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import numpy as np
import time
from contextlib import contextmanager



infcache = lru_cache(maxsize=None)


@contextmanager
def measure_time(name: str, disabled: bool = False):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if not disabled:
        print(f"{name}: {elapsed_time:.4f} seconds")


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


@dataclass
class Node:
    id: str
    layer_index: int
    token_position: int

    def to_dict(self):
        """Returns a dictionary representation of the node that can be serialized to JSON.

        Args:
            exclude_attrs: List of attribute names to exclude from the representation
        """
        exclude_attrs = ["output_vector", "input_vector", "list_contributions"]
        repr_dict = {}
        for key, value in self.__dict__.items():
            if key in exclude_attrs:
                continue

            if isinstance(value, torch.Tensor):
                temp = value.detach().cpu().to(dtype=torch.float32).numpy().tolist()
                if isinstance(temp, list):
                    repr_dict[key] = [
                        round(x, 4) if isinstance(x, float) else x for x in temp
                    ]
                else:
                    repr_dict[key] = round(temp, 4)
            else:
                repr_dict[key] = value

        repr_dict["node_type"] = self.__class__.__name__
        return repr_dict


@dataclass
class QueueElement:
    source: Node
    weight: float
    parent: Optional["QueueElement"] = None

    def __hash__(self):
        return hash((self.source.id, self.weight, hash(self.parent)))

    @property
    @infcache
    def contribution(self):
        if self.parent is None:
            return self.weight
        else:
            return self.weight * self.parent.contribution

    @property
    @infcache
    def key(self):
        key = self.contribution ** (1 / len(self.sequence))
        if isinstance(self.source, IntermediateNode):
            key *= abs(self.source.activation)
        return -key

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.key == other.key

    @property
    @infcache
    def sequence(self):
        own = [self]
        if self.parent is None:
            return own
        else:
            return own + self.parent.sequence


@dataclass
class Contribution:
    source: Node
    target: Node
    contribution: float
    vector: Optional[Tensor] = None

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "contribution": self.contribution,
        }


@dataclass
class Path:
    contributions: list[Contribution]

    @property
    def total_contribution(self):
        return sum([contribution.contribution for contribution in self.contributions])

    @property
    def id(self):
        return "->".join(
            [contribution.source.id for contribution in self.contributions]
        )

    def to_dict(self):
        return {
            "contributions": [
                contribution.to_dict() for contribution in self.contributions
            ],
            "total_contribution": self.total_contribution,
        }


@dataclass
class AttentionContribution(Contribution):
    head: int = 0


@dataclass
class OutputNode(Node):
    token_str: str
    probability: float
    logit: float
    input_vector: Tensor


@dataclass
class InputNode(Node):
    token_str: str
    output_vector: Tensor


@dataclass
class IntermediateNode(Node):
    feature_index: int
    activation: Tensor
    input_vector: Tensor
    output_vector: Tensor


@dataclass
class AttentionNode(Node):
    head: int
    source_token_position: int
    input_vector: Tensor
    output_vector: Tensor


@dataclass
class SkipNode(Node):
    output_vector: Tensor


@dataclass
class ErrorNode(Node):
    output_vector: Tensor


@dataclass
class Edge:
    @property
    def id(self):
        return f"{self.source.id} -> {self.target.id}"

    source: Node
    target: Node
    weight: float

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "weight": round(self.weight, 4),
        }


class AttributionGraph:

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        transcoders: list[torch.nn.Module],
        input_ids: Tensor,
        attention_patterns: list,
        first_ln: dict,
        pre_first_ln: dict,
        second_ln: dict,
        pre_second_ln: dict,
        input_norm: dict,
        output_norm: dict,
        attn_values: dict,
        transcoder_activations: dict,
        errors: dict,
        skip_connections: dict,
        logits: Tensor,
        last_layer_activations: Tensor,
        W_skip: list[Tensor],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transcoders = transcoders
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.input_ids = input_ids
        self.attention_patterns = attention_patterns
        self.first_ln = first_ln
        self.pre_first_ln = pre_first_ln
        self.second_ln = second_ln
        self.pre_second_ln = pre_second_ln
        self.attn_values = attn_values
        self.transcoder_activations = transcoder_activations
        self.errors = errors
        self.skip_connections = skip_connections
        self.logits = logits
        self.W_skip = [ws.detach().to(torch.bfloat16) for ws in W_skip]
        self.last_layer_activations = last_layer_activations
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.paths: list[Path] = []
        # I want to have a 2D array, where the first dimension is the layer,
        # the second is the token position and then each of these can have a list of nodes
        self.nodes_by_layer_and_token: dict[int, dict[int, list[Node]]] = {}

    def save_graph(self):
        # node_edges = {}
        # for path in self.paths:
        #     for contribution in path.contributions:
        #         if contribution.source.id == contribution.target.id:
        #             continue
        #         edge_id = f"{contribution.source.id} -> {contribution.target.id}"
        #         if edge_id not in node_edges:
        #             node_edges[edge_id] = 0
        #         node_edges[edge_id] += contribution.contribution
        # for edge, weight in node_edges.items():
        #     source,target = edge.split(" -> ")
        #     edge_id = f"{source} -> {target}"
        #     edge = Edge(id=edge_id, source=self.nodes[source], target=self.nodes[target], weight=weight)
        #     self.edges[edge_id] = edge

        dict_repr = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            # "paths": [path.to_dict() for path in self.paths]
        }
        with open("attribution_graph.json", "w") as f:
            json.dump(dict_repr, f)

    def initialize_graph(self):
        num_layers = len(self.transcoders)
        seq_len = self.input_ids.shape[0]
        self.nodes_by_layer_and_token = {
            layer: {token: [] for token in range(seq_len)}
            for layer in range(num_layers)
        }

        # Start by creating the input nodes
        for i in range(0, self.input_ids.shape[0]):
            embedding = self.model.model.embed_tokens.weight[self.input_ids[i]]
            input_node = InputNode(
                id=f"input_{i}",
                token_position=i,
                token_str=self.tokenizer.decode([self.input_ids[i]]),
                output_vector=embedding.to(dtype=torch.bfloat16),
                layer_index=-1,
            )
            self.nodes[input_node.id] = input_node
            self.nodes_by_layer_and_token[0][i].append(input_node)

        self.activation_indices_tensors = {}
        # Create the intermediate nodes
        for key in self.transcoder_activations:
            activations_tensor, indices_tensor = self.transcoder_activations[key]
            # TODO: this assumes that the keys are model.layers.i
            layer_index = int(key.split(".")[2])
            hookpoint = key

            self.activation_indices_tensors[layer_index] = dict(
                hookpoint=hookpoint,
                activations=activations_tensor,
                indices=indices_tensor,
            )

            for token_position, (top_acts, top_indices) in enumerate(
                zip(activations_tensor, indices_tensor)
            ):

                for act, index in zip(top_acts, top_indices):

                    decoder_direction = self.transcoders[hookpoint].W_dec[index, :]
                    encoder_direction = self.transcoders[hookpoint].encoder.weight[
                        index, :
                    ]
                    intermediate_node = IntermediateNode(
                        id=f"intermediate_{token_position}_{layer_index}_{index}",
                        layer_index=layer_index,
                        feature_index=index,
                        token_position=token_position,
                        activation=float(act),
                        input_vector=encoder_direction.to(dtype=torch.bfloat16),
                        output_vector=decoder_direction.to(dtype=torch.bfloat16),
                    )
                    self.nodes[intermediate_node.id] = intermediate_node
                    self.nodes_by_layer_and_token[layer_index][token_position].append(
                        intermediate_node
                    )
                # Create the error and skip nodes
                error = self.errors[key][token_position]
                error_node = ErrorNode(
                    id=f"error_{token_position}_{layer_index}",
                    layer_index=layer_index,
                    token_position=token_position,
                    output_vector=error.to(dtype=torch.bfloat16),
                )
                self.nodes_by_layer_and_token[layer_index][token_position].append(
                    error_node
                )
                self.nodes[error_node.id] = error_node
        # Create the output node
        # Top 10 logits
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.logits[0, -1, :], dim=0)
        top_10_indices = torch.argsort(probabilities, descending=True)[:10]
        top_10_probabilities = probabilities[top_10_indices]
        total_probability = 0
        for i in range(10):
            before_gradient = self.logits[0, -1, top_10_indices[i]] - torch.mean(
                self.logits[0, -1, :]
            )
            before_gradient.backward(retain_graph=True)
            gradient = self.last_layer_activations.grad
            assert gradient is not None
            self.last_layer_activations.grad = None
            # gradient = self.logits[0,-1,top_10_indices[i]].expand(gradient.shape)
            # vector = self.model.lm_head.weight[top_10_indices[i]]#-torch.mean(self.model.lm_head.weight,dim=0)
            output_node = OutputNode(
                id=f"output_{seq_len-1}_{i}",
                token_position=seq_len - 1,
                token_str=self.tokenizer.decode([top_10_indices[i]]),
                probability=top_10_probabilities[i].item(),
                logit=self.logits[0, -1, top_10_indices[i]].item(),
                input_vector=gradient[0, -1].to(dtype=torch.bfloat16),
                layer_index=len(self.transcoders),
            )
            self.nodes[output_node.id] = output_node
            total_probability += top_10_probabilities[i]
            if total_probability > 0.95:
                break

    def compute_weighted_attention_head_contribution(self):
        weighted_attention_head_contribution = []
        # We can store attention contribution instead of all the patterns and values
        OVs = []
        # inspired by get_attn_head_contribs in transcoder_circuits/transcoder_circuits/circuit_analysis.py
        for layer_index in range(len(self.attention_patterns)):
            # batch size, n_heads, seq_len, seq_len
            attention_pattern = self.attention_patterns[layer_index]
            # batch size, n_heads, seq_len, d_head
            attn_values = self.attn_values[f"model.layers.{layer_index}"].to(
                attention_pattern.device
            )
            # batch size, d_model d_model
            wO = self.model.model.layers[layer_index].self_attn.o_proj.weight.to(
                attention_pattern.device
            )
            # reshape to be n_heads d_head d_model
            wO = wO.reshape(attn_values.shape[1], -1, wO.shape[1])
            # batch head dst src, batch head src d_head -> batch head dst src d_head
            # this is different from TransformerLens order
            values_weighted_by_pattern = torch.einsum(
                "b h d s, b h s f -> b h d s f", attention_pattern, attn_values
            )

            # batch head dst src d_head, head d_head d_model -> batch head dst src d_model
            weighted_by_wO = torch.einsum(
                "b h d s f, h f m -> b h d s m", values_weighted_by_pattern, wO
            )
            weighted_attention_head_contribution.append(weighted_by_wO)
            wV = self.model.model.layers[layer_index].self_attn.v_proj.weight.to(
                attention_pattern.device
            )
            # TODO: this is 3 for smoLLM but we should get the correct number of key value heads
            wV = torch.repeat_interleave(wV, 3, dim=0)
            # reshape to be n_heads d_model d_head
            wV = wV.reshape(attn_values.shape[1], wV.shape[1], -1)

            # head d_head d_model, head d_model d_head -> head d_model d_model
            OV = torch.einsum("h f m, h n f -> h n m", wO, wV)
            OVs.append(OV)

        self.weighted_attention_head_contribution = weighted_attention_head_contribution
        self.OVs = OVs

    def layer_norm_constant(
        self,
        vector: Tensor,
        layer_index: int,
        token_position: int,
        is_ln2: bool = False,
    ) -> Tensor:
        if is_ln2:
            ln = self.second_ln[f"model.layers.{layer_index}.post_attention_layernorm"][
                0, token_position
            ]
            pre_ln = self.pre_second_ln[
                f"model.layers.{layer_index}.post_attention_layernorm"
            ][0, token_position]
        else:
            ln = self.first_ln[f"model.layers.{layer_index}"][0, token_position]
            pre_ln = self.pre_first_ln[f"model.layers.{layer_index}"][0, token_position]
        # like in transformer circuits
        # if torch.dot(vector, pre_ln) == 0:
        # return torch.tensor(0.)
        vector = vector.float()
        ln = ln.to(vector)
        pre_ln = pre_ln.to(vector)
        return torch.nan_to_num(torch.dot(vector, ln) / torch.dot(vector, pre_ln))
        # return torch.norm(ln)/torch.norm(pre_ln)
        # TODO: should it be torch.norm(ln)/torch.norm(pre_ln)?

    def compute_mlp_node_contribution(
        self,
        node: Node,
        target_node: Node,
        vector: Tensor,
    ) -> Contribution:
        if isinstance(node, IntermediateNode):
            activation = node.activation
        else:
            activation = 1

        dot_product = torch.dot(vector, node.output_vector)
        attribution = activation * dot_product

        return Contribution(
            source=node,
            target=target_node,
            contribution=attribution,
        )

    # def mlp_contribution(self, last_contribution:Contribution, layer_index:int) -> list[Contribution]:

    #     contribution_direction = last_contribution.vector

    #     # the previous source is the new target
    #     token_position = last_contribution.source.token_position
    #     target_node = last_contribution.source
    def mlp_contribution(
        self, contribution_vector: torch.Tensor, layer_index: int
    ) -> list[Contribution]:

        #     contribution_direction = last_contribution.vector

        #     # the previous source is the new target
        #     token_position = last_contribution.source.token_position
        #     target_node = last_contribution.source
        # get all the nodes that are in the same token position and layer
        contributions = []

        for token_position in range(contribution_vector.shape[-2]):
            contribution_direction = contribution_vector[..., token_position, :]
            # TODO
            while contribution_direction.ndim > 1:
                contribution_direction = contribution_direction[0]
            nodes = self.nodes_by_layer_and_token[layer_index][token_position]
            # filter out the attention nodes and the embedding node
            nodes = [
                node
                for node in nodes
                # if not isinstance(node, AttentionNode)
                # and not isinstance(node, InputNode)
                if isinstance(node, ErrorNode)
            ]
            # get the contribution of each node
            for node in tqdm(nodes, desc="Computing Node contributions", disable=True):
                new_contribution = self.compute_mlp_node_contribution(
                    node,
                    None,
                    contribution_direction,
                )
                contributions.append(new_contribution)

        acts = self.activation_indices_tensors[layer_index]
        activations, indices = acts["activations"], acts["indices"]
        while activations.ndim > 2:
            activations = activations[0]
            indices = indices[0]
        contribution = contribution_vector
        while contribution.ndim > 2:
            contribution = contribution[0]
        hookpoint = acts["hookpoint"]
        w_dec = self.transcoders[hookpoint].W_dec
        similarities = torch.einsum(
            "...fd,...d->...f",
            w_dec[indices]
            # * activations[..., None]
            , contribution)
        time_idx, feature_idx = torch.nonzero(similarities.abs() >= 1e-2, as_tuple=True)
        time_idx, feature_idx = time_idx.tolist(), feature_idx.tolist()
        sims = similarities[time_idx, feature_idx].tolist()
        for t, f, s in zip(time_idx, feature_idx, sims):
            source = self.nodes_by_layer_and_token[layer_index][t][f]
            # if abs(s) > 2 and contribution.norm(dim=-1).abs().max() < 2:
            #     print(contribution.norm(dim=-1))
            #     print(self.compute_mlp_node_contribution(source, None, contribution[t]).contribution, s)
            contributions.append(Contribution(
                source=source,
                target=None,
                contribution=s
            ))
        return [
            replace(contribution, contribution=float(contribution.contribution))
            for contribution in contributions
        ]
        # return [
        #     replace(contribution, contribution=float(contribution.contribution))
        #     for contribution in contributions
        #     if abs(contribution.contribution) >= 1e-4
        # ]

    def backward(
        self,
        # batch size, seq_len, d_model
        vector: Tensor,
        layer_index: int,
    ):
        # batch size, n_heads, seq_len, seq_len
        attention_pattern = self.attention_patterns[layer_index]
        # d_model d_model
        wO = self.model.model.layers[layer_index].self_attn.o_proj.weight.to(
            attention_pattern.device
        )
        # reshape to be n_heads d_head d_model
        wO = wO.reshape(attention_pattern.shape[1], -1, wO.shape[1])
        # batch size, n_heads, seq_len, d_head
        attn_post_value_gradient = torch.einsum("b s d, h v d -> b h s v", vector, wO)
        # batch size, n_heads, seq_len, d_head
        attn_pre_value_gradient = torch.einsum(
            "b h y v, b h y x -> b h x v", attn_post_value_gradient, attention_pattern
        )

        wV = self.model.model.layers[layer_index].self_attn.v_proj.weight.to(
            attention_pattern.device
        )
        wQ = self.model.model.layers[layer_index].self_attn.q_proj.weight.to(
            attention_pattern.device
        )
        # TODO repeat for GQA
        wV = torch.repeat_interleave(wV, wQ.shape[0] // wV.shape[0], dim=0)
        # reshape to be n_heads d_model d_head
        wV = wV.reshape(attention_pattern.shape[1], -1, wV.shape[1])
        # batch size, seq_len, d_model
        attn_pre_proj_gradient = torch.einsum(
            "b h s v, h v d -> b s d", attn_pre_value_gradient, wV
        )

        # attn_pre_proj_gradient = attn_pre_proj_gradient * self.layer_norm_constant(
        #     attn_pre_proj_gradient,
        #     layer_index,
        #     0,
        #     is_ln2=False
        # )

        ln = self.first_ln[f"model.layers.{layer_index}"]
        pre_ln = self.pre_first_ln[f"model.layers.{layer_index}"]
        attn_pre_proj_gradient = attn_pre_proj_gradient * torch.nan_to_num(ln / pre_ln)

        return attn_pre_proj_gradient + vector

        # batch head dst src, batch head src d_head -> batch head dst src d_head
        # # this is different from TransformerLens order
        # values_weighted_by_pattern = torch.einsum(
        #     'b h d s, b h s f -> b h d s f',
        #     attention_pattern, attn_values
        # )

        # #batch head dst src d_head, head d_head d_model -> batch head dst src d_model
        # weighted_by_wO = torch.einsum(
        #     'b h d s f, h f m -> b h d s m',
        #     values_weighted_by_pattern, wO
        # )
        # weighted_attention_head_contribution.append(weighted_by_wO)
        # wV = self.model.model.layers[layer_index].self_attn.v_proj.weight.to(attention_pattern.device)
        # # TODO: this is 3 for smoLLM but we should get the correct number of key value heads
        # wV = torch.repeat_interleave(wV,3,dim=0)
        # # reshape to be n_heads d_model d_head
        # wV = wV.reshape(attn_values.shape[1],wV.shape[1],-1)

        # #head d_head d_model, head d_model d_head -> head d_model d_model
        # OV = torch.einsum(
        #     'h f m, h n f -> h n m',
        #     wO, wV
        # )
        # OVs.append(OV)

    def compute_node_head_contribution(
        self, node: Node, target_node: Node, vector: Tensor
    ) -> Contribution:
        assert isinstance(node, AttentionNode)
        head = node.head
        token_position = node.token_position
        layer_index = node.layer_index

        # how much is the node contributing to the upstream node

        dot_product = torch.dot(vector, node.output_vector)
        attribution = dot_product

        # attention feature vector
        contribution_vector = node.input_vector @ vector
        ln_constant = self.layer_norm_constant(
            contribution_vector, layer_index, token_position, is_ln2=False
        )
        contribution_vector = contribution_vector * ln_constant
        new_contribution = AttentionContribution(
            source=node,
            target=target_node,
            contribution=attribution.item(),
            vector=contribution_vector,
            head=head,
        )

        return new_contribution

    def attention_contribution(
        self, last_contribution: Contribution, layer_index: int
    ) -> list[Contribution]:

        contribution_direction = last_contribution.vector

        # the previous source is the new target
        token_position = last_contribution.source.token_position
        target_node = last_contribution.source

        # get all the nodes that are in the same layer position and any token position
        # before the target node
        nodes_dict = self.nodes_by_layer_and_token[layer_index]
        nodes = []
        # need to include the target node token position
        for tok in range(token_position + 1):
            all_nodes = nodes_dict[tok]
            attention_nodes = [
                node for node in all_nodes if isinstance(node, AttentionNode)
            ]
            nodes.extend(attention_nodes)

        contributions = []

        weighted_attention_head_contribution = (
            self.weighted_attention_head_contribution[layer_index]
        )
        contrib = (
            weighted_attention_head_contribution[0, :, token_position, :]
            @ contribution_direction
        )
        # get the top 10 contributions,
        # TODO: we can potentially remove this part
        _, top_attn_contrib_indices_flattened = torch.topk(
            contrib.flatten(), k=min([50, len(contrib)])
        )
        top_attn_contrib_indices = np.array(
            np.unravel_index(
                top_attn_contrib_indices_flattened.cpu().numpy(), contrib.shape
            )
        ).T.tolist()
        for node in tqdm(nodes, desc="Computing Node contributions", disable=True):
            node_head = node.head
            node_token_position = node.token_position
            if [node_head, node_token_position] in top_attn_contrib_indices:
                new_contribution = self.compute_node_head_contribution(
                    node, target_node, contribution_direction
                )
                contributions.append(new_contribution)
        return contributions

    @torch.no_grad()
    @torch.autocast("cuda")
    def flow_once(self, queue: list[QueueElement] = [], taboo: set[str] = set()):

        with measure_time(
            "Finding a node to compute contributions",
            disabled=True,
        ):
            # if the queue is empty, get the output node with the highest probability
            # TODO: handle the other output nodes
            if len(queue) == 0:
                output_nodes = [
                    node for node in self.nodes.values() if isinstance(node, OutputNode)
                ]
                for highest_probability_node in sorted(
                    output_nodes, key=lambda x: x.probability, reverse=True
                ):
                    if highest_probability_node.id not in taboo:
                        break
                else:
                    print("Ran out of nodes")
                    return []
                taboo.add(highest_probability_node.id)
                influence, target_node = 1, highest_probability_node
                print("Doing output node")
                target = None
            else:
                # target, queue = queue[0], queue[1:]
                target = heappop(queue)
                influence, target_node = target.contribution, target.source
                if target_node.id in taboo:
                    print("Skipping node")
                    return queue
                print(f"Doing target: {target_node.id} with influence {influence}")
                print("Path:", [(x.source.id, x.weight) for x in target.sequence])

            # compute all the contributions
            max_layer = target_node.layer_index
            if isinstance(target_node, OutputNode):
                max_layer -= 1
            gradient = target_node.input_vector
            if gradient.ndim == 1:
                gradient = gradient.unsqueeze(0)
                gradient = gradient.repeat(self.input_ids.shape[-1], 1)
                gradient = (
                    gradient
                    * (
                        torch.arange(0, self.input_ids.shape[-1], device=gradient.device)
                        == target_node.token_position
                    )[:, None]
                )
        all_mlp_contributions = []
        with measure_time(
            f"Computing MLP contributions of node {target_node.id}",
            disabled=True,
        ):
            for layer in tqdm(
                range(max_layer, -1, -1),
                desc=f"Computing MLP contributions of node {target_node.id}",
                disable=True,
            ):
                if layer != max_layer:
                    contributions = self.mlp_contribution(gradient, layer)
                    all_mlp_contributions.extend(contributions)
                skipped = gradient @ self.W_skip[layer]
                ln = self.second_ln[f"model.layers.{layer}.post_attention_layernorm"]
                pre_ln = self.pre_second_ln[
                    f"model.layers.{layer}.post_attention_layernorm"
                ]
                skipped = skipped * torch.nan_to_num(ln / pre_ln)
                gradient = gradient + skipped
                gradient = self.backward(gradient, layer)
                # # print(layer, gradient.norm(dim=-1).mean())

        with measure_time(
            f"Computing embedding contributions of node {target_node.id}",
            disabled=True,
        ):
            # embedding contribution
            for preceding_position in range(0, target_node.token_position + 1):
                embed_node = [
                    node
                    for node in self.nodes_by_layer_and_token[0][preceding_position]
                    if isinstance(node, InputNode)
                ][0]
                contribution = torch.dot(
                    gradient[0, preceding_position, :], embed_node.output_vector
                ).item()
                embedding_contribution = Contribution(
                    source=embed_node, target=target_node, contribution=contribution
                )

                all_contributions = all_mlp_contributions + [embedding_contribution]

        with measure_time(
            f"Summarizing contributions of node {target_node.id}",
            disabled=True,
        ):
            # Make new paths using the last path
            new_sources = []
            for n_path in range(0, len(all_contributions)):
                new_contribution = all_contributions[n_path]
                new_source = new_contribution.source
                edge = Edge(
                    source=new_source,
                    target=target_node,
                    weight=new_contribution.contribution,
                )
                self.edges[edge.id] = edge
                # if path ends with input node, error node or skip node, it is finished and we don't want to add it to the queue
                if (
                    isinstance(new_source, InputNode)
                    or isinstance(new_source, ErrorNode)
                    or isinstance(new_source, SkipNode)
                ):
                    continue
                new_sources.append(
                    QueueElement(
                        source=new_source,
                        weight=abs(new_contribution.contribution),
                        parent=target,
                    )
                )

            existing = set(queue)
            new_sources = [x for x in new_sources if x not in existing]
            with measure_time("Sorting"):
                keys = [x.key for x in new_sources]
                topk_sort = np.argsort(keys)[:256]
                filtered_sources = [new_sources[i] for i in topk_sort]

        # sometimes sort by total contribution
        # if random.random() < 0.1:
        #     new_paths.sort(key=lambda x: x.total_contribution, reverse=True)
        # else:
        #     # the other times sort by the past contribution
        #     new_paths.sort(key=lambda x: abs(x.contributions[-1].contribution), reverse=True)

        with measure_time(
            f"Adding new sources to queue of node {target_node.id}",
            disabled=True,
        ):
            for new_source in filtered_sources:
                if new_source.source.id in taboo:
                    continue
                heappush(queue, new_source)
        # queue.extend(new_sources)
        # queue.sort(key=lambda x: abs(x.contribution), reverse=True)

        # sort the queue with the total contribution of each path
        return queue

    def flow(self, num_iterations: int = 100000):
        queue = []
        visited = set()
        for i in range(num_iterations):
            with measure_time(
                f"Iteration {i}",
                disabled=False,
            ):
                queue = self.flow_once(queue, visited)
                print(f"Queue has {len(queue)} paths")
            if not queue:
                print("Queue is empty")
                break
            # every 10 iterations, save the graph
            # if i % 10 == 0:
                # self.save_graph()
        self.save_graph()
