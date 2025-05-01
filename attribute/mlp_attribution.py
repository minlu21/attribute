import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from loguru import logger
from natsort import natsorted
from torch import Tensor
from tqdm.auto import tqdm, trange

from .caching import TranscodedModel, TranscodedOutputs
from .nodes import (Contribution, Edge, ErrorNode, InputNode, IntermediateNode,
                    Node, OutputNode)
from .utils import cantor, infcache, measure_time


@dataclass
class AttributionConfig:
    # name of the run
    name: str
    # ID for the model the features are from
    scan: str

    # how many target nodes to compute contributions for
    flow_steps: int = 2000
    # whether to use the softmax gradient for the output node
    # instead of the logit
    softmax_grad_type: Literal["softmax", "mean", "straight"] = "mean"

    # remove MLP edges below this threshold
    pre_filter_threshold: float = 1e-4
    # keep edges above this threshold
    edge_threshold = 1e-4
    # keep top k edges for each node
    top_k_edges: int = 256

    # always keep nodes above this threshold of influence
    node_threshold = 1e-3
    # keep per_layer_position nodes above this threshold for each layer/position pair
    secondary_threshold = 1e-5
    per_layer_position = 0
    # preserve all error/input nodes
    keep_all_error_nodes: bool = False
    keep_all_input_nodes: bool = True

    # correct for bias when saving top output logits
    use_logit_bias: bool = False


class AttributionGraph:
    def __init__(
        self,
        model: TranscodedModel,
        cache: TranscodedOutputs,
        config: AttributionConfig
    ):
        self.config = config
        self.model = model
        self.cache = cache
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.nodes_by_layer_and_token: dict[int, dict[int, list[Node]]] = {}
        self.initialize_graph()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def num_layers(self):
        return self.model.num_layers

    @property
    def seq_len(self):
        return self.cache.seq_len

    @property
    def input_ids(self):
        return self.cache.input_ids[0]

    @property
    def logits(self):
        return self.cache.logits[0]

    def save_graph(self, save_dir: os.PathLike):
        save_dir = Path(save_dir)
        logger.info("Saving graph to", save_dir)

        logger.info("Deduplicating nodes")
        dedup_node_names = set()
        for edge in self.edges.values():
            dedup_node_names.add(edge.source.id)
            dedup_node_names.add(edge.target.id)
        dedup_node_names = list(dedup_node_names)
        dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}

        logger.info("Finding adjacency matrix")
        logger.info(f"Number of nodes: {len(dedup_node_names)}")

        n_initial = len(dedup_node_names)
        error_mask = np.zeros((n_initial,))
        for i, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, ErrorNode):
                error_mask[i] = 1

        adj_matrix = np.zeros((n_initial, n_initial))
        for edge in self.edges.values():
            target_index = dedup_node_indices[edge.target.id]
            source_index = dedup_node_indices[edge.source.id]
            adj_matrix[target_index, source_index] = abs(edge.weight)
        adj_matrix /= np.maximum(1e-2, np.nan_to_num(adj_matrix.sum(axis=1, keepdims=True)))

        logger.info("Finding influence sources")
        influence_sources = np.zeros((n_initial,))
        for index, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, OutputNode):
                influence_sources[index] = node.probability

        total_error_influence = 1 - (influence_sources @ adj_matrix) @ error_mask
        logger.info(f"Completeness score: {total_error_influence:.3f}")

        logger.info("Finding influence matrix")
        if not hasattr(self, "influence"):
            identity = np.eye(n_initial)
            influence = np.linalg.inv(identity - adj_matrix) - identity
            self.influence = influence
        else:
            influence = self.influence

        total_error_influence = 1 - (influence_sources @ influence) @ error_mask
        logger.info(f"Replacement score: {total_error_influence:.3f}")

        usage = influence_sources @ influence
        logger.info(f"Top influences: {usage[np.argsort(usage)[-10:][::-1]].tolist()}")

        logger.info("Selecting nodes and edges")
        selected_nodes = [node for i, node in enumerate(dedup_node_names)
                  if self.nodes[node].node_type in ("OutputNode",)
                  + (("InputNode",) if self.config.keep_all_input_nodes else ())
                  + (("ErrorNode",) if self.config.keep_all_error_nodes else ())]
        for seq_idx in range(self.cache.input_ids.shape[-1]):
            for layer_idx in range(self.model.num_layers):
                matching_nodes = [
                    node for node in dedup_node_names
                    if self.nodes[node].layer_index == layer_idx
                    and self.nodes[node].token_position == seq_idx
                    and node not in selected_nodes
                ]
                matching_nodes.sort(key=lambda x: usage[dedup_node_indices[x]], reverse=True)
                matching_nodes = matching_nodes[:self.config.per_layer_position] + [
                    node
                    for node in matching_nodes[self.config.per_layer_position:]
                    if usage[dedup_node_indices[node]] > self.config.node_threshold
                ]
                matching_nodes = [
                    node for node in matching_nodes
                    if usage[dedup_node_indices[node]] > self.config.secondary_threshold
                ]
                selected_nodes.extend(matching_nodes)
        logger.info(f"Selected {len(selected_nodes)} nodes")

        export_nodes = []
        for n in selected_nodes:
            export_nodes.append(self.nodes[n])
        self.exported_nodes = export_nodes

        export_edges = []
        for edge in self.edges.values():
            if not (edge.source.id in selected_nodes and edge.target.id in selected_nodes):
                continue
            if abs(influence[dedup_node_indices[edge.target.id], dedup_node_indices[edge.source.id]]) < self.config.edge_threshold:
                continue
            export_edges.append(edge)
        self.exported_edges = export_edges
        logger.info(f"Selected {len(export_edges)} edges")

        logger.info("Exporting graph")
        tokens = [self.model.decode_token(i) for i in self.input_ids]
        # prefix = ["<EOT>"]
        prefix = []
        metadata = dict(
            slug=self.config.name,
            # scan="jackl-circuits-runs-1-4-sofa-v3_0",
            scan=self.config.scan,
            prompt_tokens=prefix + tokens,
            prompt="".join(tokens),
            title_prefix="",
            n_layers=self.model.num_layers,
        )

        nodes_json = [
            dict(
                feature_type=dict(
                    IntermediateNode="cross layer transcoder",
                    OutputNode="logit",
                    InputNode="embedding",
                    ErrorNode="mlp reconstruction error",
                )[node.node_type],
                layer=(str(node.layer_index)
                if node.layer_index != self.model.num_layers else "-1")
                if node.layer_index != -1 else "E",
                node_id=node.id_js,
                jsNodeId=node.id_js + "-0",
                feature=int(node.feature_index) if hasattr(node, "feature_index") else None,
                ctx_idx=node.token_position + len(prefix),
                run_idx=0, reverse_ctx_idx=0,
                clerp="" if not isinstance(node, OutputNode) else f"output: \"{node.token_str}\" (p={node.probability:.3f})",
                token_prob=0.0 if not isinstance(node, OutputNode) else node.probability,
                is_target_logit=False,
            ) for node in export_nodes
        ]

        for node in nodes_json:
            if node["feature_type"] != "cross layer transcoder":
                continue
            layer, feature = int(node["layer"]), int(node["feature"])
            idx_cantor = cantor(layer, feature)
            node["feature"] = idx_cantor

        links_json = [
            e.to_dict() for e in export_edges
        ]

        result = dict(
            metadata=metadata,
            nodes=nodes_json,
            links=links_json,
            qParams=dict(
                linkType="both",
                # # node IDs
                # pinnedIds=[],
                # # clickedId=export_nodes[0].id,
                # clickedId=None,
                # supernodes=[],
                # # x/y position pairs for supernodes, separated by spaces
                # sg_pos="",
            )
        )

        graph_data_dir = save_dir / "graph_data"
        circuit_path = graph_data_dir / f"{self.config.name}.json"
        open(circuit_path, "w").write(json.dumps(result))

        logger.info("Collecting metadata")
        metadatas = []
        own_file_index = 0
        for i, save_file in enumerate(graph_data_dir.glob("*.json")):
            metadatas.append(json.loads(save_file.read_text())["metadata"])
            if save_file.stem == self.config.name:
                own_file_index = i
        metadatas[0], metadatas[own_file_index] = metadatas[own_file_index], metadatas[0]
        open(save_dir / "data/graph-metadata.json", "w").write(json.dumps(dict(graphs=metadatas)))

    async def cache_features(self, cache_path: os.PathLike, save_dir: os.PathLike):
        module_latents = defaultdict(list)
        dead_features = set()
        for node in self.exported_nodes:
            if node.node_type == "IntermediateNode":
                layer, feature = int(node.layer_index), int(node.feature_index)
                dead_features.add((layer, feature))
                module_latents[self.model.temp_hookpoints_mlp[layer]].append(feature)
        module_latents = {k: torch.tensor(v) for k, v in module_latents.items()}

        ds = LatentDataset(
            cache_path,
            SamplerConfig(), ConstructorConfig(),
            modules=natsorted(module_latents.keys()),
            latents=module_latents,
        )

        logit_weight = self.model.logit_weight
        logit_bias = self.model.logit_bias

        bar = tqdm(total=sum(map(len, module_latents.values())))
        def process_feature(feature):
            layer_idx = int(feature.latent.module_name.split(".")[-2])
            feature_idx = feature.latent.latent_index
            dead_features.discard((layer_idx, feature_idx))
            index = cantor(layer_idx, feature_idx)

            feature_dir = save_dir / "features" / self.config.scan
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_path = feature_dir / f"{index}.json"

            if feature_path.exists():
                examples_quantiles = json.loads(feature_path.read_text())["examples_quantiles"]
            else:
                examples_quantiles = defaultdict(list)
                for example in feature.train:
                    examples_quantiles[example.quantile].append(dict(
                        is_repeated_datapoint=False,
                        train_token_index=len(example.tokens) - 1,
                        tokens=[self.model.decode_token(i) for i in example.tokens.tolist()],
                        tokens_acts_list=example.activations.tolist(),
                    ))
                examples_quantiles = [
                    dict(
                        quantile_name=f"Quantile {i}",
                        examples=examples_quantiles[i],
                    ) for i in sorted(examples_quantiles.keys())
                ]
            with torch.no_grad(), torch.autocast("cuda"):
                try:
                    logger.disable("attribute.caching")
                    dec_weight = self.model.w_dec(layer_idx)[feature_idx]
                finally:
                    logger.enable("attribute.caching")
                logits = logit_weight @ dec_weight
                if self.config.use_logit_bias:
                    logits += logit_bias
                top_logits = logits.topk(10).indices.tolist()
                bottom_logits = logits.topk(10, largest=False).indices.tolist()
            top_logits = [self.model.decode_token(i) for i in top_logits]
            bottom_logits = [self.model.decode_token(i) for i in bottom_logits]

            feature_vis = dict(
                index=index,
                examples_quantiles=examples_quantiles,
                bottom_logits=bottom_logits,
                top_logits=top_logits,
            )
            feature_path.write_text(json.dumps(feature_vis))
            bar.update(1)
            bar.refresh()

        async for feature in ds:
            process_feature(feature)
        bar.close()

        if len(dead_features) > 0:
            dead_features = list(dead_features)
            logger.info(f"Dead features: {dead_features[:10]}{'...' if len(dead_features) > 10 else ''}")

    def initialize_graph(self):
        num_layers = self.num_layers
        assert self.cache.batch_size == 1, "Batch size >1 not supported"
        input_ids = self.input_ids
        seq_len = self.seq_len
        self.nodes_by_layer_and_token = {
            layer: {token: [] for token in range(seq_len)}
            for layer in range(num_layers)
        }

        # Start by creating the input nodes
        for i in range(0, seq_len):
            embedding = self.model.embedding_weight[input_ids[i]]
            input_node = InputNode(
                id=f"input_{i}",
                token_position=i,
                token_str=self.tokenizer.decode([input_ids[i]]),
                output_vector=embedding.to(dtype=torch.bfloat16),
                layer_index=-1,
            )
            self.nodes[input_node.id] = input_node
            self.nodes_by_layer_and_token[0][i].append(input_node)

        self.activation_indices_tensors = {}
        # Create the intermediate nodes
        for layer, activations in self.cache.mlp_outputs.items():
            activations_tensor, indices_tensor = (
                activations.activation[0],
                activations.location[0],
            )

            self.activation_indices_tensors[layer] = dict(
                activations=activations_tensor,
                indices=indices_tensor,
            )

            for token_position, (top_acts, top_indices) in enumerate(
                zip(activations_tensor.tolist(), indices_tensor.tolist())
            ):
                for act, index in zip(top_acts, top_indices):
                    encoder_direction = self.model.w_enc(layer)[index]
                    intermediate_node = IntermediateNode(
                        id=f"intermediate_{token_position}_{layer}_{index}",
                        layer_index=layer,
                        feature_index=index,
                        token_position=token_position,
                        activation=float(act),
                        input_vector=encoder_direction.to(dtype=torch.bfloat16),
                    )
                    self.nodes[intermediate_node.id] = intermediate_node
                    self.nodes_by_layer_and_token[layer][token_position].append(
                        intermediate_node
                    )
                # Create the error and skip nodes
                error = activations.error[0, token_position]
                error_node = ErrorNode(
                    id=f"error_{token_position}_{layer}",
                    layer_index=layer,
                    token_position=token_position,
                    output_vector=error.to(dtype=torch.bfloat16),
                )
                self.nodes_by_layer_and_token[layer][token_position].append(
                    error_node
                )
                self.nodes[error_node.id] = error_node
        # Create the output node
        # Top 10 logits
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.logits[-1, :], dim=0)
        top_10_indices = torch.argsort(probabilities, descending=True)[:10]
        top_10_probabilities = probabilities[top_10_indices]
        total_probability = 0
        output_nodes = []
        for i in range(10):
            match self.config.softmax_grad_type:
                case "softmax":
                    # "logit_i - mean logit"; what does "mean logit" mean?
                    # i assume it is the mean according to the softmax distribution
                    # because that is the gradient of softmax ignoring the scaling factor
                    before_gradient = self.logits[-1, top_10_indices[i]] - \
                        torch.dot(self.logits[-1, :], self.logits[-1, :].softmax(dim=-1).detach())
                case "mean":
                    # https://transformer-circuits.pub/2025/attribution-graphs/methods.html
                    # #appendix-attribution-graph-computation
                    before_gradient = self.logits[-1, top_10_indices[i]] - torch.mean(self.logits[-1, :])
                case "straight":
                    before_gradient = self.logits[-1, top_10_indices[i]]
            before_gradient.backward(retain_graph=True)
            gradient = self.cache.last_layer_activations.grad
            assert gradient is not None
            self.cache.last_layer_activations.grad = None
            # gradient = self.logits[0,-1,top_10_indices[i]].expand(gradient.shape)
            # vector = self.model.lm_head.weight[top_10_indices[i]]#-torch.mean(self.model.lm_head.weight,dim=0)
            output_node = OutputNode(
                id=f"output_{seq_len-1}_{i}",
                token_position=seq_len - 1,
                token_str=self.tokenizer.decode([top_10_indices[i]]),
                probability=top_10_probabilities[i].item(),
                logit=self.logits[-1, top_10_indices[i]].item(),
                input_vector=gradient[0, -1].to(dtype=torch.bfloat16),
                layer_index=self.num_layers,
            )
            self.nodes[output_node.id] = output_node
            output_nodes.append(output_node)
            total_probability += top_10_probabilities[i]
            if total_probability > 0.95:
                break
        self.output_nodes = output_nodes

        # cleared each time we re-initialize the graph
        self.queue = NodeQueue()
        self.remaining_output_nodes = output_nodes.copy()

    def compute_mlp_node_contribution(
        self,
        node: Node,
        target_node: Node,
        vector: Tensor,
    ) -> Contribution:
        dot_product = torch.dot(vector.float(), node.output_vector.float())
        attribution = dot_product.item()

        return Contribution(
            source=node,
            target=target_node,
            contribution=attribution,
        )

    def mlp_contribution(
        self, contribution_vector: torch.Tensor, layer_index: int
    ) -> dict[str, Contribution]:
        contributions = {}

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
                if isinstance(node, ErrorNode)
            ]
            # get the contribution of each node
            for node in tqdm(nodes, desc="Computing Node contributions", disable=True):
                new_contribution = self.compute_mlp_node_contribution(
                    node,
                    None,
                    contribution_direction,
                )
                contributions[new_contribution.source.id] = new_contribution

        contribution = contribution_vector
        while contribution.ndim > 2:
            contribution = contribution[0]
        for source_layer in range(layer_index + 1):
            acts = self.activation_indices_tensors[source_layer]
            activations, indices = acts["activations"], acts["indices"]
            while activations.ndim > 2:
                activations = activations[0]
                indices = indices[0]
            w_dec = self.model.w_dec(source_layer, layer_index)
            similarities = torch.einsum(
                "...fd,...d->...f",
                w_dec[indices]
                * activations[..., None]
                , contribution)
            time_idx, feature_idx = torch.nonzero(similarities.abs() >= self.config.pre_filter_threshold, as_tuple=True)
            sims = similarities[time_idx, feature_idx].tolist()
            time_idx = time_idx.tolist()
            indexes = indices[time_idx, feature_idx].tolist()
            for t, s, i in zip(time_idx, sims, indexes):
                source_id = f"intermediate_{t}_{source_layer}_{i}"
                source = self.nodes[source_id]
                contributions[source_id] = Contribution(
                    source=source,
                    target=None,
                    contribution=s / source.activation
                )

        return {
            k: replace(contribution, contribution=float(contribution.contribution))
            for k, contribution in contributions.items()
        }

    def attn_backward(
        self,
        # batch size, seq_len, d_model
        vector: Tensor,
        layer_index: int,
    ):
        # batch size, n_heads, seq_len, seq_len
        attention_pattern = self.cache.attn_outputs[layer_index].attn_patterns
        # d_model n_heads d_head
        wO = self.model.attn_output(layer_index).to(
            attention_pattern.device
        )
        # batch size, n_heads, seq_len, d_head
        attn_post_value_gradient = torch.einsum("b s d, d h v -> b h s v", vector, wO)
        # batch size, n_heads, seq_len, d_head
        attn_pre_value_gradient = torch.einsum(
            "b h y v, b h y x -> b h x v", attn_post_value_gradient, attention_pattern
        )

        wV = self.model.attn_value(layer_index).to(
            attention_pattern.device
        )
        # batch size, seq_len, d_model
        attn_pre_proj_gradient = torch.einsum(
            "b h s v, h v d -> b s d", attn_pre_value_gradient, wV
        )
        attn_pre_proj_gradient = attn_pre_proj_gradient * self.cache.attn_outputs[layer_index].ln_factor

        return attn_pre_proj_gradient

    @torch.no_grad()
    @torch.autocast("cuda")
    def flow_once(self):
        with measure_time(
            "Finding a node to compute contributions",
            disabled=True,
        ):
            # if the queue is empty, get the output node with the highest probability
            # TODO: handle the other output nodes
            if len(self.output_nodes) > 0:
                influence, target_node = 1, self.output_nodes.pop()
                logger.debug("Starting from output node")
                target_elem = None
            else:
                if len(self.queue) == 0:
                    return False
                target_elem = self.queue.pop()
                influence, target_node = target_elem.contribution, target_elem.source
                logger.debug(f"Doing target: {target_node.id} with influence {influence}")
                logger.debug("Path:", [(x.source.id, x.weight) for x in target_elem.sequence])

            # compute all the contributions
            max_layer = target_node.layer_index
            gradient = target_node.input_vector
            if gradient.ndim == 1:
                gradient = gradient.unsqueeze(0)
                gradient = gradient.repeat(self.seq_len, 1)
                gradient = (
                    gradient
                    * (
                        torch.arange(0, self.seq_len, device=gradient.device)
                        == target_node.token_position
                    )[:, None]
                )
            if not isinstance(target_node, OutputNode):
                gradient = gradient * self.cache.mlp_outputs[max_layer].ln_factor
            else:
                # just to fix the shape (insert batch dimension),
                # the gradient is already scaled by the ln_factor
                gradient = gradient * torch.ones_like(self.cache.final_ln_factor)
        all_mlp_contributions = {}
        past_gradients = {}
        with measure_time(
            f"Computing MLP contributions of node {target_node.id}",
            disabled=True,
        ):
            for layer in tqdm(
                range(min(max_layer, self.model.num_layers - 1), -1, -1),
                desc=f"Computing MLP contributions of node {target_node.id}",
                disable=True,
            ):
                start_gradient = gradient
                # pass skip gradient (just after the MLP) to the previous layer
                past_gradients[layer] = start_gradient
                if layer != max_layer:
                    # find MLP sources on the current layer
                    mlp_contributions = self.mlp_contribution(gradient, layer)
                    for k, v in mlp_contributions.items():
                        if k in all_mlp_contributions:
                            all_mlp_contributions[k].contribution += v.contribution
                        else:
                            all_mlp_contributions[k] = v

                    # process skip weights from future layers
                    skipped = 0
                    to_delete = set()
                    for target_layer, grad in past_gradients.items():
                        try:
                            skipped += grad @ self.model.w_skip(layer, target_layer)
                        except IndexError:
                            to_delete.add(target_layer)
                    for target_layer in to_delete:
                        del past_gradients[target_layer]
                    skipped = skipped * self.cache.mlp_outputs[layer].ln_factor
                    gradient = gradient + skipped

                if self.model.parallel_attn:
                    gradient += self.attn_backward(start_gradient, layer)
                else:
                    gradient += self.attn_backward(gradient, layer)

        all_contributions = list(all_mlp_contributions.values())
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
                    gradient[0, preceding_position, :].float(), embed_node.output_vector.float()
                ).item()
                embedding_contribution = Contribution(
                    source=embed_node, target=target_node, contribution=contribution
                )

                all_contributions.append(embedding_contribution)

        with measure_time(
            f"Summarizing contributions of node {target_node.id}",
            disabled=True,
        ):
            with measure_time(
                f"Creating sources for {target_node.id}",
                disabled=True,
            ):
                # Make new paths using the last path
                new_sources = []
                for n_path in range(0, len(all_contributions)):
                    new_contribution = all_contributions[n_path]
                    new_source = new_contribution.source
                    weight = new_contribution.contribution
                    if isinstance(new_source, IntermediateNode):
                        weight *= new_source.activation
                    edge = Edge(
                        source=new_source,
                        target=target_node,
                        weight=weight,
                    )
                    self.edges[edge.id] = edge
                    # if path ends with input node or error node, it is finished and we don't want to add it to the queue
                    if (
                        isinstance(new_source, InputNode)
                        or isinstance(new_source, ErrorNode)
                    ):
                        continue
                    new_sources.append(
                        QueueElement(
                            source=new_source,
                            weight=abs(new_contribution.contribution),
                            parent=target_elem,
                        )
                    )

            self.queue.add_many(new_sources, self.config.top_k_edges)

        return len(self.queue) > 0 or target_elem is None

    def flow(self, num_iterations: Optional[int] = None):
        if num_iterations is None:
            num_iterations = self.config.flow_steps
        for i in (bar := trange(num_iterations, desc="Flowing contributions")):
            with measure_time(
                f"Iteration {i}",
                disabled=True,
            ):
                if not self.flow_once():
                    logger.debug("Queue is empty")
                    break
                logger.debug(f"Queue has {len(self.queue)} paths")
                bar.set_postfix(queue_elements=len(self.queue))


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
        key = self.contribution  # ** (1 / len(self.sequence))
        if isinstance(self.source, IntermediateNode):
            key *= abs(self.source.activation)
        return -abs(key)

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.source.id == other.source.id

    @property
    @infcache
    def sequence(self):
        own = [self]
        if self.parent is None:
            return own
        else:
            return own + self.parent.sequence


class NodeQueue:
    def __init__(self):
        self.visited = set()
        self.unvisited = {}

    def __len__(self):
        return len(self.unvisited)

    def pop(self):
        all_layers = set(x.source.layer_index for x in self.unvisited.values())
        random_layer = random.choice(list(all_layers))
        unvisited = {x: y for x, y in self.unvisited.items() if y.source.layer_index == random_layer}
        highest = min(unvisited, key=lambda x: self.unvisited[x].key)
        if highest in self.visited:
            del self.unvisited[highest]
            return self.pop()
        return self.unvisited.pop(highest)

    def add(self, node: QueueElement):
        if node.source.id in self.visited:
            return
        if node.source.id in self.unvisited:
            if node.key < self.unvisited[node.source.id].key:
                self.unvisited[node.source.id] = node
        else:
            self.unvisited[node.source.id] = node

    def add_many(self, nodes: list[QueueElement], top_k: int = float("inf")):
            with measure_time("Deduplicating", disabled=True):
                new_sources = [x for x in nodes if x.source.id not in self.visited]
            with measure_time("Creating keys to sort", disabled=True):
                keys = [float(x.key) for x in new_sources]
            with measure_time("Sorting", disabled=True):
                if len(keys) > top_k:
                    topk_sort = np.argpartition(keys, top_k)[:top_k]
                else:
                    topk_sort = np.arange(len(keys))
                filtered_sources = [new_sources[i] for i in topk_sort]
            with measure_time("Adding to queue", disabled=True):
                for source in filtered_sources:
                    self.add(source)
