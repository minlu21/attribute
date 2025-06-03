import json
import os
import random
import numpy as np
import torch
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from loguru import logger
from tqdm.auto import tqdm, trange
from jaxtyping import Float, Int, Bool, Array

from .caching import TranscodedModel, TranscodedOutputs
from .nodes import (Contribution, Edge, ErrorNode, InputNode, IntermediateNode,
                    Node, OutputNode)
from .utils import cantor, measure_time


DEBUG = os.environ.get("ATTRIBUTE_DEBUG", "0") == "1"
if DEBUG:
    def node_from_index(index: int, edge_matrix: dict, self):
        try:
            layer_idx_0, seq_idx_0, feature_idx_0 = edge_matrix["activation_matrix_indices"][index].tolist()
        except IndexError:
            err_index = index - edge_matrix["activation_matrix_indices"].shape[0]
            num_positions = self.cache.original_input_ids.shape[-1]
            layer_idx_0 = err_index // num_positions
            seq_idx_0 = err_index % num_positions
            if layer_idx_0 >= self.model.num_layers:
                token_index = err_index - (num_positions * self.model.num_layers)
                if token_index < num_positions:
                    return f"input_{token_index - 1}"
                token_index -= num_positions
                token_id = edge_matrix["logit_tokens"][token_index]
                return [node.id for node in self.output_nodes if node.token_idx == token_id][0]
            return f"error_{seq_idx_0 - 1}_{layer_idx_0}"
        seq_idx_0 = seq_idx_0 - 1
        if (seq_idx_0, layer_idx_0, feature_idx_0) not in self.intermediate_nodes:
            print("MISSING NODE:", (seq_idx_0, layer_idx_0, feature_idx_0))
            return None
        return self.intermediate_nodes[(seq_idx_0, layer_idx_0, feature_idx_0)].id


@dataclass
class AttributionConfig:
    # name of the run
    name: str
    # ID for the model the features are from
    scan: str
    # seed for random number generator
    seed: int = 42

    # how many target nodes to compute contributions for
    flow_steps: int = 5000
    # batch size for MLP attribution
    batch_size: int = 8
    # whether to use the softmax gradient for the output node
    # instead of the logit
    softmax_grad_type: Literal["softmax", "mean", "straight"] = "mean"

    # remove features that are this dense
    filter_high_freq_early: float = 0.01

    # remove MLP edges below this threshold
    pre_filter_threshold: float = 1e-5 if not DEBUG else 0.0
    # keep edges that make up this fraction of the total influence
    edge_cum_threshold: float = 0.98
    # keep top k edges for each node
    top_k_edges: int = 32 if not DEBUG else 16384
    # whether to multiply by activation strength before computing influence
    influence_anthropic: bool = True
    # whether to compute influence on the GPU
    influence_gpu: bool = True
    # keep nodes that make up this fraction of the total influence
    node_cum_threshold: float = 0.8
    # keep per_layer_position nodes above this threshold for each layer/position pair
    secondary_threshold = 1e-5
    per_layer_position = 0
    # preserve all error/input nodes
    keep_all_error_nodes: bool = False
    keep_all_input_nodes: bool = True
    # debugging: use all nodes as targets
    use_all_targets: bool = DEBUG

    # correct for bias when saving top output logits
    use_logit_bias: bool = False

    use_self_explanation: bool = False
    selfe_min_strength: float = 2.0
    selfe_max_strength: float = 8.0
    self_sim_layer: int = 15
    selfe_n: int = 64
    selfe_pick: int = 5
    selfe_n_tokens: int = 8


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
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

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

    def adjacency_matrix(self, normalize: bool = True, absolute: bool = True):
        logger.info("Deduplicating nodes")
        dedup_node_names = set()
        for edge in self.edges.values():
            dedup_node_names.add(edge.source.id)
            dedup_node_names.add(edge.target.id)
        dedup_node_names = list(dedup_node_names)
        dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}

        n_initial = len(dedup_node_names)
        activation_sources = np.ones((n_initial,))
        for index, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, IntermediateNode):
                activation_sources[index] = node.activation

        logger.info("Finding adjacency matrix")
        logger.info(f"Number of nodes: {len(dedup_node_names)}")

        adj_matrix = np.zeros((n_initial, n_initial))
        for edge in self.edges.values():
            target_index = dedup_node_indices[edge.target.id]
            source_index = dedup_node_indices[edge.source.id]
            weight = edge.weight
            if absolute:
                weight = abs(weight)
            adj_matrix[target_index, source_index] = weight * activation_sources[source_index]

        if DEBUG:
            edge_matrix = torch.load("../circuit-replicate/edge_matrix.pt")
            # pruned_matrix = torch.load("../circuit-replicate/pruned_graph.pt")
            # edge_scores = pruned_matrix["normalized_pruned"].cpu()
            # edge_scores = pruned_matrix["pruned_matrix"].cpu()
            edge_scores = edge_matrix["edge_matrix"].abs().cpu()

            from tqdm import trange
            from collections import Counter
            indices_adj = []
            indices_edge = []
            n_unused = 0
            counter = Counter()
            for i in trange(edge_scores.shape[0]):
                source_id = node_from_index(i, edge_matrix, self)
                if source_id is None:
                    print("UNUSED:", i, i - edge_matrix["activation_matrix_indices"].shape[0])
                    n_unused += 1
                    continue
                if source_id not in dedup_node_indices:
                    print("NOT INCLUDED:", source_id, edge_scores[:, i].abs().max(), edge_scores[i, :].abs().max())
                    n_unused += 1
                    continue
                counter[dedup_node_indices[source_id]] += 1
                indices_adj.append(dedup_node_indices[source_id])
                indices_edge.append(i)
                if edge_scores[:, i].abs().max() < 1e-5:
                    print("ZERO COLUMN:", source_id, edge_scores[:, i].abs().max())
                if edge_scores[i, :].abs().max() < 1e-5:
                    print("ZERO ROW:", source_id, edge_scores[i, :].abs().max())
                # for j in range(edge_scores.shape[1]):
                #     source_id = node_from_index(i)
                #     target_id = node_from_index(j)
                #     if source_id is None or target_id is None:
                #         continue
                #     if source_id not in dedup_node_indices or target_id not in dedup_node_indices:
                #         continue
                #     adj_matrix[dedup_node_indices[target_id], dedup_node_indices[source_id]] = edge_scores[i, j]
            self.indices_adj = indices_adj
            print(counter.most_common(10))
            print(f"{n_unused / edge_scores.shape[0]:.3f}")
            indices_adj = np.array(indices_adj)
            indices_edge = np.array(indices_edge)
            # adj_matrix *= 0
            # adj_matrix[indices_adj[:, None], indices_adj[None, :]] = edge_scores[indices_edge[:, None], indices_edge[None, :]]
            # adj_matrix[indices_adj][:, indices_adj] = edge_scores[indices_edge][:, indices_edge]
            adj_mgrid = np.meshgrid(indices_adj, indices_adj)
            edge_mgrid = np.meshgrid(indices_edge, indices_edge)
            adj_matrix[adj_mgrid[0].flatten(), adj_mgrid[1].flatten()] = edge_scores[edge_mgrid[0].flatten(), edge_mgrid[1].flatten()]
            for i in range(adj_matrix.shape[0]):
                if np.abs(adj_matrix[i]).sum() < 1e-5:
                    print("ZERO ROW:", i, np.abs(adj_matrix[i]).sum())
                    print(dedup_node_names[i], i in indices_adj, edge_scores[indices_edge[indices_adj == i], :].sum())
                    print(self.nodes[dedup_node_names[i]])
            # exit()

        if normalize:
            adj_matrix /= np.maximum(1e-2, np.nan_to_num(adj_matrix.sum(axis=1, keepdims=True)))

        return adj_matrix, dedup_node_names, activation_sources

    def make_latent_dataset(self, cache_path: os.PathLike, module_latents: dict[str, torch.Tensor]):
        if self.model.pre_ln_hook and module_latents:
            module_latents = {k.replace(".mlp", f".{self.model.mlp_layernorm_name}"): v for k, v in module_latents.items()}
        return LatentDataset(
            cache_path,
            SamplerConfig(n_examples_train=10, train_type="top", n_examples_test=0),
            ConstructorConfig(center_examples=False, example_ctx_len=16, n_non_activating=0),
            modules=list(module_latents.keys()) if module_latents else None,
            latents=module_latents,
        )

    @torch.inference_mode()
    def find_influence(self, adj_matrix: np.ndarray):
        n_initial = adj_matrix.shape[0]
        if self.config.influence_anthropic:
            adj_matrix = np.abs(adj_matrix)
            if self.config.influence_gpu:
                adj_gpu = torch.from_numpy(adj_matrix).to(self.model.device)
                adj_gpu /= adj_gpu.sum(dim=1, keepdim=True).clamp(min=1e-10)
                identity = torch.eye(n_initial, device=self.model.device)
                influence = torch.inverse(identity - adj_gpu) - identity
                influence = influence.cpu().numpy()
                adj_matrix = adj_gpu.cpu().numpy()
            else:
                normalizer = adj_matrix.sum(axis=1, keepdims=True)
                normalizer = np.where(normalizer > 0, normalizer, 1)
                adj_matrix = adj_matrix / normalizer
                influence = np.linalg.inv(np.eye(n_initial) - adj_matrix) - np.eye(n_initial)
        else:
            influence = np.linalg.inv(np.eye(n_initial) - adj_matrix) - np.eye(n_initial)
            influence = np.abs(influence)
            influence = influence / np.maximum(1e-2, influence.sum(axis=1, keepdims=True))
            adj_matrix = np.abs(adj_matrix)
            adj_matrix = adj_matrix / np.maximum(1e-2, adj_matrix.sum(axis=1, keepdims=True))
        return influence, adj_matrix

    def save_graph(self, save_dir: os.PathLike):
        save_dir = Path(save_dir)
        logger.info("Saving graph to", save_dir)

        adj_matrix, dedup_node_names, activation_sources = self.adjacency_matrix(absolute=True, normalize=False)
        dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}

        n_initial = len(dedup_node_names)
        error_mask = np.zeros((n_initial,))
        for i, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, ErrorNode):
                error_mask[i] = 1

        logger.info("Finding influence sources")
        influence_sources = np.zeros((n_initial,))
        for index, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, OutputNode):
                influence_sources[index] = node.probability

        if DEBUG:
            edge_matrix = torch.load("../circuit-replicate/edge_matrix.pt")
            logit_weights = edge_matrix["logit_weights"].tolist()
            # logit_weights = edge_matrix["logit_w"].tolist()
            logit_tokens = edge_matrix["logit_tokens"].tolist()
            for index, node in enumerate(dedup_node_names):
                node = self.nodes[node]
                if isinstance(node, OutputNode):
                    output_id = node.token_idx
                    logit_idx = logit_tokens.index(output_id)
                    influence_sources[index] = logit_weights[logit_idx]

        logger.info("Finding influence matrix")
        original_adj_matrix = adj_matrix.copy()

        influence, adj_matrix = self.find_influence(adj_matrix)

        total_error_influence = 1 - (influence_sources @ adj_matrix) @ error_mask
        logger.info(f"Completeness score (unpruned): {total_error_influence:.3f}")

        total_error_influence = 1 - (influence_sources @ influence) @ error_mask
        logger.info(f"Replacement score (unpruned): {total_error_influence:.3f}")

        usage = influence_sources @ influence
        logger.info(f"Top influences: {usage[np.argsort(usage)[-10:][::-1]].tolist()}")

        if DEBUG:
            edge_matrix = torch.load("../circuit-replicate/edge_matrix.pt")
            logit_weights = edge_matrix["logit_weights"]

            def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
                normalized = matrix.abs()
                return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)

            def compute_influence(A):
                # Calculate total influence using matrix inverse (I - A)^-1 - I
                identity = torch.eye(A.shape[0], device=A.device)
                B = torch.inverse(identity - A) - identity
                return logit_weights @ B.to(logit_weights.device)

            # Calculate node influence and apply threshold
            # node_influence = compute_influence(normalize_matrix(edge_matrix["edge_matrix"]))
            pruned_matrix = torch.load("../circuit-replicate/pruned_graph.pt")
            node_influence = pruned_matrix["node_influence"].cpu().tolist()
            # act_mat_vals = edge_matrix["activation_matrix_values"].cpu().tolist()
            # unnormalized = edge_matrix["edge_matrix"].abs().cpu().numpy()
            # node_sums = unnormalized.sum(axis=1)
            # inf_sums = np.abs(original_adj_matrix).sum(axis=1)

            xs = []
            ys = []
            # for i, (layer_idx, seq_idx, feature_idx) in enumerate(edge_matrix["activation_matrix_indices"].tolist()):
            #     seq_idx = seq_idx - 1
            #     try:
            #         node_id = self.intermediate_nodes[(seq_idx, layer_idx, feature_idx)].id
            #     except KeyError:
            #         print("(usage) MISSING NODE:", (seq_idx, layer_idx, feature_idx), act_mat_vals[i])
            #         continue
            #     try:
            #         dedup_node_indices[node_id]
            #     except KeyError:
            #         print("(usage) NOT INCLUDED:", node_id, self.nodes[node_id].activation)
            #         continue
            counted = set(dedup_node_indices.keys())
            for i in range(len(node_influence)):
                node_id = node_from_index(i, edge_matrix, self)
                if node_id is None or node_id not in dedup_node_indices:
                    continue
                counted.remove(node_id)
                x, y = node_influence[i], usage[dedup_node_indices[node_id]]
                # x, y = node_sums[i], inf_sums[dedup_node_indices[node_id]]
                # xs.append(node_influence[i])
                xs.append(x)
                # xs.append(act_mat_vals[i])
                ys.append(y)
                # ys.append(usage[dedup_node_indices[node_id]])
                # ys.append(activation_sources[dedup_node_indices[node_id]])
                # if y <= 0:
                #     print(node_id, layer_idx, seq_idx, feature_idx, x, y, act_mat_vals[i], activation_sources[dedup_node_indices[node_id]], np.abs(original_adj_matrix[:, dedup_node_indices[node_id]]).sum(), dedup_node_indices[node_id] in self.indices_adj)
                #     for j, v in enumerate(unnormalized[i]):
                #         if v > 0:
                #             node_src = node_from_index(j, edge_matrix, self)
                #             if node_src is not None and node_src in dedup_node_indices:
                #                 print("", node_src, node_src in dedup_node_indices, v, original_adj_matrix[dedup_node_indices[node_id], dedup_node_indices[node_src]], dedup_node_indices[node_src] in self.indices_adj)
            print("Not counted:", counted)
            from matplotlib import pyplot as plt
            # plt.xscale("log")
            # plt.yscale("log")
            plt.scatter(xs, ys, s=1)
            plt.plot([1e-5, 1], [1e-5, 1], color="black")
            plt.xlabel("Theirs")
            plt.ylabel("Ours")
            plt.title(f"R^2: {np.corrcoef(xs, ys)[0, 1]**2:.3f}")
            plt.savefig("results/influence_vs_usage.png")
            plt.close()
        # exit()

        logger.info("Selecting nodes and edges")
        selected_nodes = [node for i, node in enumerate(dedup_node_names)
                  if self.nodes[node].node_type in ("OutputNode",)
                  + (("InputNode",) if self.config.keep_all_input_nodes else ())
                  + (("ErrorNode",) if self.config.keep_all_error_nodes else ())]

        sorted_usage = np.sort(usage)[::-1]
        cumsum_usage = np.cumsum(sorted_usage)
        cumsum_usage = cumsum_usage / cumsum_usage[-1]
        node_threshold = sorted_usage[np.searchsorted(cumsum_usage, self.config.node_cum_threshold)]
        logger.info(f"Node threshold: {node_threshold}")

        for seq_idx in range(self.cache.input_ids.shape[-1]):
            for layer_idx in range(self.num_layers):
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
                    if usage[dedup_node_indices[node]] > node_threshold
                ]
                matching_nodes = [
                    node for node in matching_nodes
                    if usage[dedup_node_indices[node]] > self.config.secondary_threshold
                ]
                selected_nodes.extend(matching_nodes)

        logger.info(f"Selected {len(selected_nodes)} nodes")

        filtered_mask = np.zeros((len(dedup_node_names),), dtype=bool)
        for node in selected_nodes:
            filtered_mask[dedup_node_indices[node]] = 1
        # for node in dedup_node_names:
        #     if node.startswith("error"):
        #         filtered_mask[dedup_node_indices[node]] = 1
        filtered_index = np.cumsum(filtered_mask) - 1

        filtered_adj_matrix = original_adj_matrix[filtered_mask][:, filtered_mask].copy()
        # orig_filtered_adj_matrix = filtered_adj_matrix
        filtered_influence, filtered_adj_matrix = self.find_influence(filtered_adj_matrix)
        filtered_node_influence = influence_sources[filtered_mask] @ filtered_influence
        filtered_influence = filtered_adj_matrix * (filtered_node_influence + influence_sources[filtered_mask])[None, :]

        total_pruned_influence = float(1 - (influence_sources[filtered_mask] @ filtered_adj_matrix) @ error_mask[filtered_mask])
        logger.info(f"Completeness score: {total_pruned_influence:.3f}")
        total_pruned_influence = float(1 - filtered_node_influence @ error_mask[filtered_mask])
        logger.info(f"Replacement score: {total_pruned_influence:.3f}")

        selected_edge_matrix = np.sort(filtered_influence.flatten())[::-1]
        edge_cumsum = np.cumsum(selected_edge_matrix)
        edge_cumsum = edge_cumsum / edge_cumsum[-1]

        if DEBUG:
            from matplotlib import pyplot as plt
            plt.loglog(1 - edge_cumsum)
            plt.plot([0, 10000], [0.02, 0.02], "k--")
            plt.savefig("results/edge_cumsum.png")
            plt.close()

        if DEBUG:
            edge_matrix = torch.load("../circuit-replicate/edge_matrix.pt")
            pruned_matrix = torch.load("../circuit-replicate/pruned_graph.pt")
            mask = pruned_matrix["node_mask"].tolist()
            total = 0
            matching = 0
            accounted_for = filtered_mask.copy()
            for i, v in enumerate(mask):
                if v:
                    total += 1
                    node = node_from_index(i, edge_matrix, self)
                    if node is None:
                        print("NO NODE:", i)
                        continue
                    if node in selected_nodes:
                        matching += 1
                        accounted_for[dedup_node_indices[node]] = 0
                    else:
                        print("WRONG NODE:", i, node, node in selected_nodes, usage[dedup_node_indices[node]], adj_matrix[:, dedup_node_indices[node]].sum())
            for i, v in enumerate(accounted_for):
                if v:
                    print("not accounted for:", i, dedup_node_names[i], usage[i])

            print(f"{matching / total:.3f}", matching, total)

            # edge_scores = edge_matrix["edge_matrix"].abs().cpu()
            # edge_scores = pruned_matrix["normalized"].cpu()
            edge_scores = pruned_matrix["normalized_pruned"].cpu()
            # edge_scores = pruned_matrix["pruned_matrix"].cpu()
            # edge_scores = pruned_matrix["edge_scores"].cpu()
            xs = []
            ys = []
            colors = []
            from tqdm import trange
            for i in trange(edge_scores.shape[0]):
                source_id = node_from_index(i, edge_matrix, self)
                if source_id is None or source_id not in dedup_node_indices:
                    continue
                for j in range(edge_scores.shape[1]):
                    score = float(edge_scores[i, j])
                    if score == 0:
                        continue
                    target_id = node_from_index(j, edge_matrix, self)
                    if target_id is None or target_id not in dedup_node_indices:
                        continue
                    source_node = self.nodes[source_id]
                    target_node = self.nodes[target_id]

                    # if source_node.layer_index - target_node.layer_index > 1:
                    #     continue
                    # if source_node.token_position != target_node.token_position:
                    #     continue

                    source_idx = filtered_index[dedup_node_indices[source_id]]
                    target_idx = filtered_index[dedup_node_indices[target_id]]

                    if filtered_mask[dedup_node_indices[source_id]] == 0 or filtered_mask[dedup_node_indices[target_id]] == 0:
                        continue
                    x, y = score, filtered_adj_matrix[source_idx, target_idx]
                    # x, y = np.abs(score), orig_filtered_adj_matrix[source_idx, target_idx]
                    if abs(x - y) > 1e-1:
                        print("VERY WRONG:", source_id, target_id, x, y)
                    xs.append(x)
                    # ys.append(influence[dedup_node_indices[source_id], dedup_node_indices[target_id]])
                    # ys.append(np.abs(original_adj_matrix[dedup_node_indices[source_id], dedup_node_indices[target_id]]))
                    # ys.append(adj_matrix[dedup_node_indices[source_id], dedup_node_indices[target_id]])
                    # ys.append(filtered_influence[source_idx, target_idx])
                    ys.append(y)
                    # ys.append(orig_filtered_adj_matrix[source_idx, target_idx])
                    # ys.append(np.abs(original_adj_matrix[dedup_node_indices[source_id], dedup_node_indices[target_id]]))# * activation_sources[dedup_node_indices[target_id]])
                    # print()
                    # print(i, j, score)
                    # print(source_idx, target_idx)
                    # print(filtered_influence[source_idx, target_idx])
                    if isinstance(source_node, IntermediateNode) and isinstance(target_node, IntermediateNode):
                        colors.append("green")
                    elif isinstance(target_node, ErrorNode):
                        colors.append("red")
                    elif isinstance(target_node, InputNode):
                        colors.append("blue")
                    elif isinstance(target_node, IntermediateNode):
                        colors.append("yellow")
                    else:
                        colors.append("black")
            from matplotlib import pyplot as plt
            plt.scatter(xs, ys, s=1, c=colors)
            y_min, y_max = min(ys), max(ys)
            plt.plot([y_min, y_max], [y_min, y_max], color="black")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Theirs")
            plt.ylabel("Ours")
            plt.savefig("results/influence_vs_score.png")
            plt.close()
            print("AAAAAAA")
        edge_threshold = selected_edge_matrix[np.searchsorted(edge_cumsum, self.config.edge_cum_threshold)]
        logger.info(f"Edge threshold: {edge_threshold}")

        export_edges = []
        has_incoming = set()
        has_outgoing = set()
        for edge in self.edges.values():
            if not (edge.source.id in selected_nodes and edge.target.id in selected_nodes):
                continue
            if abs(
                filtered_influence[filtered_index[dedup_node_indices[edge.target.id]],
                                   filtered_index[dedup_node_indices[edge.source.id]]]
                ) < edge_threshold:
                continue
            has_outgoing.add(edge.source.id)
            has_incoming.add(edge.target.id)
            export_edges.append(edge)
        self.exported_edges = export_edges
        logger.info(f"Selected {len(export_edges)} edges")

        export_nodes = []
        for n in selected_nodes:
            if isinstance(self.nodes[n], IntermediateNode):
                if n not in has_incoming or n not in has_outgoing:
                    continue
            if isinstance(self.nodes[n], ErrorNode):
                if n not in has_outgoing:
                    continue
            export_nodes.append(self.nodes[n])
        self.exported_nodes = export_nodes
        logger.info(f"Final number of nodes: {len(export_nodes)}")

        logger.info("Exporting graph")
        export_offset = self.cache.original_input_ids.shape[-1] - self.cache.input_ids.shape[-1]
        input_ids = self.cache.original_input_ids[0].tolist()
        tokens = [self.model.decode_token(i) for i in input_ids]
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
            node_threshold=node_threshold,
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
                ctx_idx=node.token_position + len(prefix) + export_offset,
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
        graph_data_dir.mkdir(parents=True, exist_ok=True)
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
        (save_dir / "data").mkdir(parents=True, exist_ok=True)
        open(save_dir / "data/graph-metadata.json", "w").write(json.dumps(dict(graphs=metadatas)))

        return circuit_path

    def get_dense_features(self, cache_path: os.PathLike):
        cache_path = Path(cache_path)
        dense_features = set()
        if not cache_path.exists() or not len(list(cache_path.glob("*"))) or not self.config.filter_high_freq_early:
            logger.warning("Skipping dense feature detection because cache does not exist or filter_high_freq_early is 0")
            self.dense_features = dense_features
            return
        dense_cache_path = cache_path.parent / "dense_features.json"
        if dense_cache_path.exists():
            dense = json.loads(dense_cache_path.read_text())
        else:
            dense = []
            ds = self.make_latent_dataset(cache_path, None)
            for buf in tqdm(ds.buffers, desc="Finding dense features"):
                module = buf.module_path
                all_features, _, all_tokens = buf.load()
                all_features = all_features[..., 2]

                # unique, counts = torch.unique(all_features, return_counts=True)
                # freqs = counts / all_tokens.numel()
                # too_high = freqs > self.config.filter_high_freq_early

                all_tokens, all_features = all_tokens.numpy(), all_features.numpy()
                unique, counts = np.unique(all_features, return_counts=True)
                freqs = counts / all_tokens.size
                too_high = freqs > self.config.filter_high_freq_early

                # dead_features[module].extend(unique[too_high].tolist())
                if self.model.pre_ln_hook:
                    module = module.replace(f".{self.model.mlp_layernorm_name}", ".mlp")
                layer_idx = self.model.temp_hookpoints_mlp.index(module)
                feature_names = [[layer_idx, feature] for feature in unique[too_high].tolist()]
                dense.extend(feature_names)
            open(dense_cache_path, "w").write(json.dumps(dense))

        self.dense_features = set((pos, *feature) for pos in range(self.seq_len) for feature in dense)

    def cache_features(self, cache_path: os.PathLike, save_dir: os.PathLike):
        cache_path = Path(cache_path)
        save_dir = Path(save_dir)
        logit_weight = self.model.logit_weight
        logit_bias = self.model.logit_bias
        for node in tqdm(self.exported_nodes, desc="Caching features"):
            if node.node_type == "IntermediateNode":
                feature_dir = save_dir / "features" / self.config.scan

                layer, feature = int(node.layer_index), int(node.feature_index)

                with torch.no_grad(), torch.autocast("cuda"):
                    try:
                        logger.disable("attribute.caching")
                        dec_weight = self.model.w_dec_i(layer, feature)
                    finally:
                        logger.enable("attribute.caching")
                    logits = logit_weight @ dec_weight
                    del dec_weight
                    if self.config.use_logit_bias:
                        logits += logit_bias
                    top_logits = logits.topk(10).indices.tolist()
                    bottom_logits = logits.topk(10, largest=False).indices.tolist()
                top_logits = [self.model.decode_token(i) for i in top_logits]
                bottom_logits = [self.model.decode_token(i) for i in bottom_logits]

                feature_vis = dict(
                    index=cantor(layer, feature),
                    bottom_logits=bottom_logits,
                    top_logits=top_logits,
                )
                feature_dir.mkdir(parents=True, exist_ok=True)
                feature_path = feature_dir / f"{cantor(layer, feature)}.json"
                if feature_path.exists():
                    try:
                        feature_vis = json.loads(feature_path.read_text()) | feature_vis
                    except json.JSONDecodeError:
                        pass
                feature_path.write_text(json.dumps(feature_vis))

    def cache_self_explanations(self, cache_path: os.PathLike, save_dir: os.PathLike):
        if not self.config.use_self_explanation:
            return
        cache_path = Path(cache_path)
        save_dir = Path(save_dir)
        for node in tqdm(self.exported_nodes, desc="Caching self-explanations"):
            if node.node_type == "IntermediateNode":
                explanation = self.self_explain_feature(node)
                feature_dir = save_dir / "features" / self.config.scan
                feature_dir.mkdir(parents=True, exist_ok=True)
                feature_path = feature_dir / f"{cantor(node.layer_index, node.feature_index)}.json"
                if feature_path.exists():
                    explanation = json.loads(feature_path.read_text()) | explanation
                feature_path.write_text(json.dumps(explanation))

    @torch.inference_mode()
    def self_explain_feature(self, node: IntermediateNode):
        layer, feature = node.layer_index, node.feature_index
        w_dec = self.model.w_dec_i(layer, feature, use_skip=True)
        w_enc = self.model.w_enc_i(layer, feature)
        strengths = torch.linspace(self.config.selfe_min_strength, self.config.selfe_max_strength, self.config.selfe_n, device=self.model.device)
        w_dec = w_dec[None, :] * strengths[:, None]
        w_enc = w_enc[None, :] * strengths[:, None]
        dec_explanations = self.self_explain_generate(w_dec, min_strength=self.config.selfe_min_strength, max_strength=self.config.selfe_max_strength, seq_len=self.config.selfe_n_tokens)
        enc_explanations = self.self_explain_generate(w_enc, min_strength=self.config.selfe_min_strength, max_strength=self.config.selfe_max_strength, seq_len=self.config.selfe_n_tokens)
        return dict(
            self_explanation_enc=enc_explanations,
            self_explanation_dec=dec_explanations,
        )

    def self_explain_generate(self, vectors: torch.Tensor,
                              *, min_strength: float,
                              max_strength: float,
                              seq_len: int):
        prompt = "ONLINE DICTIONARY\nThe meaning of the word ? is \""
        tokenized_prompt = self.model.tokenizer.encode(prompt)
        question_mark_token = self.model.tokenizer.encode("a ?")[-1]
        token_index = tokenized_prompt.index(question_mark_token)
        emb_vec = vectors / vectors.norm(dim=-1, keepdim=True)
        batch_size = vectors.shape[0]
        strengths = torch.linspace(min_strength, max_strength, batch_size, device=self.model.device)
        emb_vec = emb_vec[None, :] * strengths[:, None]
        tokens = self.model.tokenizer([prompt] * batch_size, return_tensors="pt").to(self.model.device).input_ids
        state = (tokens, tokens, None, None, None)
        def step(state):
            all_tokens, tokens, cache, entropy, self_similarity = state
            for m in self.model.model.modules():
                m._forward_hooks = OrderedDict()

            if tokens.shape[1] != 1:
                def patch(module, input, output):
                    output[:, token_index] = emb_vec
                    return output

                self.model.embedding_module.register_forward_hook(patch)

                def collect_self_similarity(module, input, output):
                    nonlocal self_similarity
                    if isinstance(output, tuple):
                        output = output[0]
                    self_similarity = torch.nn.functional.cosine_similarity(
                        output[:, -1],
                        emb_vec,
                        dim=-1
                    )[0]

                self.model.get_layer(self.config.self_sim_layer).register_forward_hook(collect_self_similarity)

            output = self.model.model(tokens, past_key_values=cache)
            logits = output.logits

            if entropy is None:
                probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
                logprobs = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
                entropy = -torch.sum(probs * logprobs, dim=-1)

            probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            all_tokens = torch.cat([all_tokens, next_tokens], dim=1)
            return (all_tokens, next_tokens, output.past_key_values, entropy, self_similarity)

        try:
            for _ in range(seq_len):
                logger.disable("attribute")
                state = step(state)
                tokens = state[0]
        finally:
            logger.enable("attribute")
        decoded = [self.model.tokenizer.decode(seq[len(tokenized_prompt):]) for seq in tokens.tolist()]
        entropy, self_similarity = state[3:5]
        entropy = entropy - entropy.min()
        entropy = entropy / entropy.max()
        self_similarity = self_similarity - self_similarity.min()
        self_similarity = self_similarity / self_similarity.max()
        top_self_sims = torch.topk(self_similarity, k=self.config.selfe_pick).indices.tolist()
        decoded = [decoded[i] for i in top_self_sims]
        # decoded = [str((float(entropy), float(self_sim))) + decoded for entropy, self_sim, decoded in zip(entropy, self_similarity, decoded)]
        decoded = [decoded.partition('"')[0].partition("\n")[0] for decoded in decoded]
        return decoded

    async def cache_contexts(self, cache_path: os.PathLike, save_dir: os.PathLike):
        cache_path = Path(cache_path)
        save_dir = Path(save_dir)
        feature_paths = {}
        module_latents = defaultdict(list)
        dead_features = set()
        for node in self.exported_nodes:
            if node.node_type == "IntermediateNode":
                layer_idx = node.layer_index
                feature_idx = node.feature_index
                feature_dir = save_dir / "features" / self.config.scan
                feature_dir.mkdir(parents=True, exist_ok=True)
                feature_path = feature_dir / f"{cantor(layer_idx, feature_idx)}.json"
                if feature_path.exists():
                    if "examples_quantiles" in json.loads(feature_path.read_text()):
                        continue
                feature_paths[(layer_idx, feature_idx)] = feature_path
                module_latents[self.model.temp_hookpoints_mlp[layer_idx]].append(feature_idx)
                dead_features.add((layer_idx, feature_idx))

        module_latents = {k: torch.tensor(v) for k, v in module_latents.items()}
        module_latents = {k: v[torch.argsort(v)] for k, v in module_latents.items()}

        ds = self.make_latent_dataset(cache_path, module_latents)

        bar = tqdm(total=sum(map(len, module_latents.values())))
        def process_feature(feature):
            layer_idx = int(feature.latent.module_name.split(".")[-2])
            feature_idx = feature.latent.latent_index
            dead_features.discard((layer_idx, feature_idx))

            feature_path = feature_paths[(layer_idx, feature_idx)]

            feature_vis = json.loads(feature_path.read_text())
            examples_quantiles = feature_vis.get("examples_quantiles", None)

            if examples_quantiles is None:
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
            feature_vis["examples_quantiles"] = examples_quantiles

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
        logger.info(f"Initializing graph with {num_layers} layers")
        seq_len = self.seq_len
        self.nodes_by_layer_and_token = {
            layer: {token: [] for token in range(seq_len)}
            for layer in range(num_layers)
        }

        offset = self.cache.mlp_outputs[0].source_activation.shape[1] - seq_len

        # Start by creating the input nodes
        og_input_ids = self.cache.original_input_ids[0]
        for i in range(-offset, seq_len):
            embedding = self.model.embedding_weight[og_input_ids[i]]
            input_node = InputNode(
                id=f"input_{i}",
                token_position=i,
                token_idx=og_input_ids[i],
                token_str=self.tokenizer.decode([og_input_ids[i]]),
                output_vector=embedding.to(dtype=torch.bfloat16),
                layer_index=-1,
            )
            self.nodes[input_node.id] = input_node
            if i >= 0:
                self.nodes_by_layer_and_token[0][i].append(input_node)

        self.activation_indices_tensors = {}
        self.intermediate_nodes = {}
        self.intermediate_nodes_k = {}
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
                for k_index, (act, index) in enumerate(zip(top_acts, top_indices)):
                    intermediate_node = IntermediateNode(
                        id=f"intermediate_{token_position}_{layer}_{index}",
                        layer_index=layer,
                        feature_index=index,
                        token_position=token_position,
                        activation=float(act),
                        input_vector=None,
                    )
                    self.nodes[intermediate_node.id] = intermediate_node
                    self.nodes_by_layer_and_token[layer][token_position].append(
                        intermediate_node
                    )
                    self.intermediate_nodes_k[(token_position, layer, k_index)] = intermediate_node
                    self.intermediate_nodes[(token_position, layer, index)] = intermediate_node
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
                token_idx=top_10_indices[i],
                token_str=self.tokenizer.decode([top_10_indices[i]]),
                probability=top_10_probabilities[i].item(),
                logit=self.logits[-1, top_10_indices[i]].item(),
                input_vector=gradient[0, -1].to(dtype=torch.bfloat16),
                layer_index=self.num_layers,
                logit_idx=i,
            )
            self.nodes[output_node.id] = output_node
            output_nodes.append(output_node)
            total_probability += top_10_probabilities[i]
            if total_probability > 0.95:
                break
        self.output_nodes = output_nodes

        # cleared each time we re-initialize the graph
        self.queue = AttributionQueue(self.cache, self)
        self.remaining_output_nodes = output_nodes.copy()

        self.dead_features = set()

    @torch.autocast("cuda")
    def flow_once(self):
        true_seq_len = self.cache.input_ids.shape[1]
        fake_seq_len = self.cache.mlp_outputs[0].source_activation.shape[1]
        offset = fake_seq_len - true_seq_len

        gradient = 0
        mlp_source = False
        # if the queue is empty, get the output node with the highest probability
        # TODO: handle the other output nodes
        if len(self.remaining_output_nodes) > 0:
            target_node = self.remaining_output_nodes.pop()
            influences, target_nodes = [target_node.probability], [target_node]
            logger.debug("Starting from output node")
        else:
            if len(self.queue) == 0:
                return False
            influences, max_layer, target_indices = self.queue.pop_n(self.config.batch_size)
            mlp_source = True
            list_target_indices = target_indices.tolist()
            target_nodes = [self.intermediate_nodes_k[i, max_layer, j] for i, j in list_target_indices]

        # compute all the contributions
        if isinstance(target_nodes[0], OutputNode):
            max_layer = target_nodes[0].layer_index
            gradient = target_nodes[0].input_vector
            target_graph_node = self.cache.last_layer_activations[0, target_nodes[0].token_position + offset]
            max_mlp_layer = self.model.num_layers
        elif mlp_source:
            target_graph_node = self.cache.mlp_outputs[max_layer].activation
            gradient = None
            batch_idx = torch.arange(len(target_indices))
            gradient = torch.zeros_like(target_graph_node)
            gradient[batch_idx, target_indices[:, 0], target_indices[:, 1]] += 1
            max_mlp_layer = max_layer
        else:
            raise ValueError

        backward_to = [self.cache.first_layer_activations] + [
            node
            for i in range(max_mlp_layer)
            for node in (
                self.cache.mlp_outputs[i].source_activation,
                self.cache.mlp_outputs[i].source_error,
            )
        ]
        gradients = torch.autograd.grad(
            [target_graph_node],
            backward_to,
            [gradient],
            retain_graph=True,
        )

        if DEBUG and False:
            edge_matrix = torch.load("../circuit-replicate/edge_matrix.pt")
            mlp_indices = edge_matrix["activation_matrix_indices"].tolist()

        with measure_time("Summarizing contributions of node", disabled=True), torch.no_grad():
            for batch_idx, (target_node, influence) in enumerate(zip(target_nodes, influences)):
                all_contributions = []
                for seq_idx in range(-offset, target_node.token_position + 1):
                    contribution = gradients[0][batch_idx][seq_idx + offset] @ backward_to[0][batch_idx][seq_idx + offset]
                    input_node_name = f"input_{seq_idx}"
                    source = self.nodes[input_node_name]
                    all_contributions.append(Contribution(
                        source=source,
                        target=target_node,
                        contribution=contribution.to(device="cpu", non_blocking=True),
                    ))

                error_grads = {}
                # if DEBUG and isinstance(target_node, IntermediateNode) and random.random() < 0.01 and target_node.layer_index > 0:
                if False:
                    layer_idx = target_node.layer_index - 1
                    if 1:
                        error_index = 2 + layer_idx * 2
                        error_grad, error_val = gradients[error_index], backward_to[error_index]

                        error_grad_ = error_grad[batch_idx, -true_seq_len:]
                        try:
                            target_index = mlp_indices.index([target_node.layer_index, target_node.token_position + 1, target_node.feature_index])
                        except ValueError:
                            continue
                        cache_path = Path("../circuit-replicate/grads")
                        from natsort import natsorted
                        u = True
                        for pt_file in natsorted(cache_path.glob("*.pt")):
                            # grads_blocks.0.mlp.hook_out_3691_3698.pt
                            import re
                            try:
                                source_layer_, start_target, end_target = re.match(r"grads_blocks\.([0-9]+)\.mlp\.hook_out_([0-9]+)_([0-9]+)\.pt", pt_file.name).groups()
                            except AttributeError:
                                continue

                            source_layer_ = int(source_layer_)
                            start_target = int(start_target)
                            end_target = int(end_target)
                            if layer_idx != source_layer_:
                                continue
                            if target_index < start_target or target_index >= end_target:
                                continue
                            break
                        else:
                            u = False
                        if u:
                            pt_file = torch.load(pt_file, weights_only=False)
                            index_of_target = target_index - start_target

                            gradients_ = pt_file["grads"][index_of_target]
                            error_grads[layer_idx] = gradients_

                            xs = error_grad_.detach().cpu().flatten().float().numpy()
                            ys = gradients_[1:].detach().cpu().flatten().float().numpy()
                            from matplotlib import pyplot as plt
                            plt.scatter(xs, ys)
                            os.makedirs("results/grads", exist_ok=True)
                            plt.title(f"Target: {target_node.layer_index}, {target_node.token_position}, {target_node.feature_index}, Source: {source_layer_}")
                            plt.savefig(f"results/grads/{target_node.layer_index}_{target_node.token_position}_{target_node.feature_index}.png")
                            plt.close()

                for layer_idx in range(max_mlp_layer):
                    error_index = 2 + layer_idx * 2
                    error_grad, error_val = gradients[error_index], backward_to[error_index]

                    for seq_idx in range(target_node.token_position + 1):
                        error_contribution = error_grad[batch_idx, -true_seq_len:][seq_idx] @ error_val[batch_idx, -true_seq_len:][seq_idx]
                        source = self.nodes[f"error_{seq_idx}_{layer_idx}"]
                        assert source.token_position <= target_node.token_position, f"{source.token_position} <= {target_node.token_position}"
                        all_contributions.append(Contribution(
                            source=source,
                            target=target_node,
                            contribution=error_contribution.to(device="cpu", non_blocking=True),
                        ))

                for layer_idx in range(max_mlp_layer):
                    mlp_index = 1 + layer_idx * 2
                    mlp_grad = gradients[mlp_index][batch_idx, -true_seq_len:]
                    edge = ((mlp_grad * (mlp_grad.abs() >= self.config.pre_filter_threshold)).abs() + (1e-8 if self.config.use_all_targets else 0)) * (influence if not self.config.use_all_targets else 1) * self.cache.mlp_outputs[layer_idx].activation[0]
                    self.queue.layers[layer_idx].contributions += edge
                    n_elem = min(self.config.top_k_edges, (edge > 0).sum().item())
                    if n_elem == 0:
                        continue
                    edge_indices = torch.topk(edge.flatten(), n_elem).indices
                    edge_seq, edge_k = edge_indices // edge.shape[-1], edge_indices % edge.shape[-1]
                    # if DEBUG and layer_idx == 15:
                    #     print("!!!!", edge_seq, edge_k)
                    #     exit()
                    edge_feat = self.cache.mlp_outputs[layer_idx].location[batch_idx, -true_seq_len:].flatten()[edge_indices]
                    edge_val = mlp_grad[edge_seq, edge_k]
                    self.queue.layers[layer_idx].edge_source_seq.append(edge_seq.to("cpu", non_blocking=True))
                    self.queue.layers[layer_idx].edge_source_idx.append(edge_feat.to("cpu", non_blocking=True))
                    self.queue.layers[layer_idx].edge_target_layer.append(target_node.layer_index)
                    self.queue.layers[layer_idx].edge_target_seq.append(target_node.token_position)
                    self.queue.layers[layer_idx].edge_target_idx.append(target_node.feature_index if isinstance(target_node, IntermediateNode) else (-1 - target_node.logit_idx))
                    self.queue.layers[layer_idx].edge_weight.append(edge_val.to("cpu", non_blocking=True))

                # Make new paths using the last path
                for n_path in range(0, len(all_contributions)):
                    new_contribution = all_contributions[n_path]
                    new_source = new_contribution.source
                    assert new_source.token_position <= target_node.token_position, f"{new_source.token_position} <= {target_node.token_position}"
                    weight = float(new_contribution.contribution)
                    edge = Edge(
                        source=new_source,
                        target=target_node,
                        weight=float(weight),
                    )
                    self.edges[edge.id] = edge

        if mlp_source:
            for source_layer, source_seq, source_idx, target_layer, target_seq, target_idx, weight in self.queue.purge():
                source = self.intermediate_nodes[(source_seq, source_layer, source_idx)]
                if target_idx < 0:
                    target = self.output_nodes[-1 - target_idx]
                else:
                    target = self.intermediate_nodes[(target_seq, target_layer, target_idx)]
                if DEBUG and isinstance(source, IntermediateNode) and source.layer_index == 15 and source.token_position < 5:
                    print("SOURCE:",source)
                # if isinstance(target, IntermediateNode) and target.layer_index == 15:
                #     print(target)
                #     1/0
                edge = Edge(
                    source=source,
                    target=target,
                    weight=weight
                )
                self.edges[edge.id] = edge

        return len(self.queue) > 0 or not mlp_source

    def flow(self, num_iterations: Optional[int] = None):
        if num_iterations is None:
            num_iterations = self.config.flow_steps // self.config.batch_size
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
class AttributionLayer:
    visited: Bool[Array, "seq_len k"]
    contributions: Float[Array, "seq_len k"]
    edge_source_seq: list[Int[Array, "..."]]
    edge_source_idx: list[Int[Array, "..."]]
    edge_target_layer: list[Int[Array, "..."]]
    edge_target_seq: list[Int[Array, "..."]]
    edge_target_idx: list[Int[Array, "..."]]
    edge_weight: list[Float[Array, "..."]]


class AttributionQueue:
    def __init__(self, cache: TranscodedOutputs, graph: AttributionGraph):
        self.graph = graph
        self.cache = cache
        self.layers = {
            layer: AttributionLayer(
                visited=torch.zeros(output.activation.shape[1:], device=cache.input_ids.device, dtype=torch.bool),
                contributions=torch.zeros(output.activation.shape[1:], device=cache.input_ids.device, dtype=torch.float32),
                edge_source_seq=[],
                edge_source_idx=[],
                edge_target_layer=[],
                edge_target_seq=[],
                edge_target_idx=[],
                edge_weight=[],
            )
            for layer, output in cache.mlp_outputs.items()
        }

    @torch.no_grad()
    def contribution_not_visited(self, layer: int, multiply_by_activation: bool = False):
        return (self.layers[layer].contributions * ~self.layers[layer].visited * (self.cache.mlp_outputs[layer].activation[0] if multiply_by_activation else 1)).flatten()

    @torch.no_grad()
    def pop_n(self, n: int):
        layer = random.choice(list(layer for layer in self.layers.keys() if self.contribution_not_visited(layer).sum().item() > 0))
        contribution_not_visited = self.contribution_not_visited(layer)
        n_to_visit = min(n, (contribution_not_visited > 0).sum().item())
        contribution, visited = torch.topk(contribution_not_visited, n_to_visit)
        k = self.layers[layer].contributions.shape[-1]
        visited_seq, visited_k = visited // k, visited % k
        self.layers[layer].visited[visited_seq, visited_k] = True
        return contribution.tolist(), layer, torch.stack([visited_seq, visited_k], dim=-1)

    def __len__(self):
        return sum((self.contribution_not_visited(layer) > 0).sum().item() for layer in self.layers.keys())

    def purge(self):
        for layer_idx, layer in self.layers.items():
            while len(layer.edge_source_seq) > 0:
                source_seqs, source_idces, weights = (
                    layer.edge_source_seq.pop().tolist(),
                    layer.edge_source_idx.pop().tolist(),
                    layer.edge_weight.pop().tolist(),
                )
                target_layer, target_seq, target_idx = (
                    int(layer.edge_target_layer.pop()),
                    int(layer.edge_target_seq.pop()),
                    int(layer.edge_target_idx.pop()),
                )
                for source_seq, source_idx, weight in zip(source_seqs, source_idces, weights):
                    if weight == 0:
                        continue
                    yield layer_idx, source_seq, source_idx, target_layer, target_seq, target_idx, weight
