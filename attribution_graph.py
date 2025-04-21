from dataclasses import dataclass
import torch
from torch import Tensor
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import numpy as np
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
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
        exclude_attrs = ["output_vector", "input_vector","list_contributions"]
        repr_dict = {}
        for key, value in self.__dict__.items():
            if key in exclude_attrs:
                continue

            if isinstance(value, torch.Tensor):
                temp = value.detach().cpu().to(dtype=torch.float32).numpy().tolist()
                if isinstance(temp, list):
                    repr_dict[key] = [round(x, 4) if isinstance(x, float) else x for x in temp]
                else:
                    repr_dict[key] = round(temp, 4)
            else:
                repr_dict[key] = value

        repr_dict["node_type"] = self.__class__.__name__
        return repr_dict


@dataclass
class Contribution:
    source: Node
    target: Node
    vector: Tensor
    contribution: float
    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "contribution": self.contribution
        }


@dataclass
class Path:
    contributions: list[Contribution]
    @property
    def total_contribution(self):
        # return sum([contribution.contribution for contribution in self.contributions])
        result = 1
        for contribution in self.contributions:
            result *= contribution.contribution
        return result
    @property
    def id(self):
        return "->".join([contribution.source.id for contribution in self.contributions])
    def to_dict(self):
        return {
            "contributions": [contribution.to_dict() for contribution in self.contributions],
            "total_contribution": self.total_contribution
        }


@dataclass
class AttentionContribution(Contribution):
    head: int

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
    id: str
    source: Node
    target: Node
    weight: float
    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "weight": round(self.weight,4)
        }




class AttributionGraph:

    def __init__(self,
                 model:torch.nn.Module,
                 tokenizer:AutoTokenizer,
                 transcoders:list[torch.nn.Module],
                 input_ids:Tensor,
                 attention_patterns:list,
                 first_ln:dict,
                 pre_first_ln:dict,
                 second_ln:dict,
                 pre_second_ln:dict,
                 input_norm:dict,
                 output_norm:dict,
                 attn_values:dict,
                 transcoder_activations:dict,
                 errors:dict,
                 skip_connections:dict,
                 logits:Tensor,
                 last_layer_activations:Tensor):
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
        self.last_layer_activations = last_layer_activations
        self.nodes: dict[str,Node] = {}
        self.edges: dict[str,Edge] = {}
        self.paths: list[Path] = []
        # I want to have a 2D array, where the first dimension is the layer,
        # the second is the token position and then each of these can have a list of nodes
        self.nodes_by_layer_and_token : dict[int,dict[int,list[Node]]] = {}

    def save_graph(self):
        node_edges = {}
        for path in self.paths:
            for contribution in path.contributions:
                if contribution.source.id == contribution.target.id:
                    continue
                edge_id = f"{contribution.source.id} -> {contribution.target.id}"
                if edge_id not in node_edges:
                    node_edges[edge_id] = 0
                node_edges[edge_id] += contribution.contribution
        for edge, weight in node_edges.items():
            source,target = edge.split(" -> ")
            edge_id = f"{source} -> {target}"
            edge = Edge(id=edge_id, source=self.nodes[source], target=self.nodes[target], weight=weight)
            self.edges[edge_id] = edge


        dict_repr = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "paths": [path.to_dict() for path in self.paths]
        }
        with open("attribution_graph.json", "w") as f:
            json.dump(dict_repr, f)


    def initialize_graph(self):
        num_layers = len(self.transcoders)
        seq_len = self.input_ids.shape[0]
        self.nodes_by_layer_and_token = {layer: {token: [] for token in range(seq_len)} for layer in range(num_layers)}

        # Start by creating the input nodes
        for i in range(0,self.input_ids.shape[0]):
            embedding =self.model.model.embed_tokens.weight[self.input_ids[i]]
            input_node = InputNode(id=f"input_{i}",
                                   token_position=i,
                                   token_str=self.tokenizer.decode([self.input_ids[i]]),
                                   output_vector=embedding.to(dtype=torch.bfloat16),
                                   layer_index=-1)
            self.nodes[input_node.id] = input_node
            self.nodes_by_layer_and_token[0][i].append(input_node)
        # Create the intermediate nodes
        for key in self.transcoder_activations:
            activations_tensor, indices_tensor = self.transcoder_activations[key]
            #TODO: this assumes that the keys are model.layers.i
            layer_index = int(key.split(".")[2])
            hookpoint = key

            for token_position, (top_acts, top_indices) in enumerate(zip(activations_tensor, indices_tensor)):

                for act, index in zip(top_acts, top_indices):

                    decoder_direction = self.transcoders[hookpoint].W_dec[index,:]
                    encoder_direction = self.transcoders[hookpoint].encoder.weight[index,:]
                    intermediate_node = IntermediateNode(id=f"intermediate_{token_position}_{layer_index}_{index}",
                                                        layer_index=layer_index,
                                                        feature_index=index,
                                                        token_position=token_position,
                                                        activation=act,
                                                        input_vector=encoder_direction.to(dtype=torch.bfloat16),
                                                        output_vector=decoder_direction.to(dtype=torch.bfloat16))
                    self.nodes[intermediate_node.id] = intermediate_node
                    self.nodes_by_layer_and_token[layer_index][token_position].append(intermediate_node)
                # Create the error and skip nodes
                error = self.errors[key][token_position]
                skip = self.skip_connections[key][token_position]
                error_node = ErrorNode(id=f"error_{token_position}_{layer_index}",
                                    layer_index=layer_index,
                                    token_position=token_position,
                                    output_vector=error.to(dtype=torch.bfloat16))
                self.nodes_by_layer_and_token[layer_index][token_position].append(error_node)
                self.nodes[error_node.id] = error_node
                skip_node = SkipNode(id=f"skip_{token_position}_{layer_index}",
                                    layer_index=layer_index,
                                    token_position=token_position,
                                    output_vector=skip.to(dtype=torch.bfloat16))
                self.nodes[skip_node.id] = skip_node
                self.nodes_by_layer_and_token[layer_index][token_position].append(skip_node)
        # Create the attention nodes
        self.compute_weighted_attention_head_contribution()
        for layer_index in range(len(self.attention_patterns)):
            for head in range(len(self.attention_patterns[layer_index][0])):
                for target_token_position in range(seq_len):
                    for source_token_position in range(target_token_position+1):
                        # score * OV
                        input_vector = (self.OVs[layer_index][head]*
                                        self.attention_patterns[layer_index][0,head,target_token_position,source_token_position]
                                        ).to(dtype=torch.bfloat16)
                        # score * OV * resid
                        output_vector = self.weighted_attention_head_contribution[layer_index]
                        output_vector = output_vector[0,head,target_token_position,source_token_position].to(dtype=torch.bfloat16)
                        attention_node = AttentionNode(id=f"attention_{target_token_position}_{layer_index}_{source_token_position}_{head}",
                                                      layer_index=layer_index,
                                                      head=head,
                                                      token_position=target_token_position,
                                                      source_token_position=source_token_position,
                                                      input_vector=input_vector,
                                                      output_vector=output_vector)
                        self.nodes[attention_node.id] = attention_node
                        self.nodes_by_layer_and_token[layer_index][source_token_position].append(attention_node)

        # Create the output node
        # Top 10 logits
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.logits[0,-1,:],dim=0)
        top_10_indices = torch.argsort(probabilities,descending=True)[:10]
        top_10_probabilities = probabilities[top_10_indices]
        total_probability = 0
        for i in range(10):
            before_gradient = (self.logits[0,-1,top_10_indices[i]]-torch.mean(self.logits[0,-1,:]))
            before_gradient.backward(retain_graph=True)
            gradient = self.last_layer_activations.grad
            assert gradient is not None
            self.last_layer_activations.grad = None
            #gradient = self.logits[0,-1,top_10_indices[i]].expand(gradient.shape)
            #vector = self.model.lm_head.weight[top_10_indices[i]]#-torch.mean(self.model.lm_head.weight,dim=0)
            output_node = OutputNode(id=f"output_{seq_len-1}_{i}",
                                     token_position=seq_len-1,
                                     token_str=self.tokenizer.decode([top_10_indices[i]]),
                                     probability=top_10_probabilities[i].item(),
                                     logit=self.logits[0,-1,top_10_indices[i]].item(),
                                     input_vector=gradient[0,-1].to(dtype=torch.bfloat16),
                                     layer_index=len(self.transcoders))
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
            attn_values = self.attn_values[f"model.layers.{layer_index}"].to(attention_pattern.device)
            # batch size, d_model d_model
            wO = self.model.model.layers[layer_index].self_attn.o_proj.weight.to(attention_pattern.device)
            # reshape to be n_heads d_head d_model
            wO = wO.reshape(attn_values.shape[1],-1,wO.shape[1])
            #batch head dst src, batch head src d_head -> batch head dst src d_head
            # this is different from TransformerLens order
            values_weighted_by_pattern = torch.einsum(
                'b h d s, b h s f -> b h d s f',
                attention_pattern, attn_values
            )

            #batch head dst src d_head, head d_head d_model -> batch head dst src d_model
            weighted_by_wO = torch.einsum(
                'b h d s f, h f m -> b h d s m',
                values_weighted_by_pattern, wO
            )
            weighted_attention_head_contribution.append(weighted_by_wO)
            wV = self.model.model.layers[layer_index].self_attn.v_proj.weight.to(attention_pattern.device)
            # TODO: this is 3 for smoLLM but we should get the correct number of key value heads
            wV = torch.repeat_interleave(wV,3,dim=0)
            # reshape to be n_heads d_model d_head
            wV = wV.reshape(attn_values.shape[1],wV.shape[1],-1)

            #head d_head d_model, head d_model d_head -> head d_model d_model
            OV = torch.einsum(
                'h f m, h n f -> h n m',
                wO, wV
            )
            OVs.append(OV)

        self.weighted_attention_head_contribution = weighted_attention_head_contribution
        self.OVs = OVs


    def layer_norm_constant(self,vector:Tensor,layer_index:int,token_position:int,is_ln2:bool=False) -> Tensor:
        if is_ln2:
            ln = self.second_ln[f"model.layers.{layer_index}.post_attention_layernorm"][0,token_position]
            pre_ln = self.pre_second_ln[f"model.layers.{layer_index}.post_attention_layernorm"][0,token_position]
        else:
            ln = self.first_ln[f"model.layers.{layer_index}"][0,token_position]
            pre_ln = self.pre_first_ln[f"model.layers.{layer_index}"][0,token_position]
        # like in transformer circuits
        if torch.dot(vector, pre_ln) == 0:
            return torch.tensor(0.)
        return torch.dot(vector, ln)/torch.dot(vector, pre_ln)
        #return torch.norm(ln)/torch.norm(pre_ln)
        # TODO: should it be torch.norm(ln)/torch.norm(pre_ln)?

    def compute_mlp_node_contribution(self, node:Node,
                                      target_node:Node,
                                      vector:Tensor,
                                      ) -> Contribution:
                layer_index = node.layer_index
                token_position = node.token_position
                if isinstance(node, IntermediateNode) :
                    activation = node.activation
                else:
                    activation = 1

                dot_product = torch.dot(vector, node.output_vector)
                attribution = activation * dot_product
                # apply layer norm
                if isinstance(node,IntermediateNode):
                    contribution_vector = dot_product * node.input_vector
                    ln_constant = self.layer_norm_constant(contribution_vector,
                                                           layer_index,
                                                           token_position, is_ln2=True)
                    contribution_vector = contribution_vector * ln_constant

                else:
                    contribution_vector = torch.zeros_like(node.output_vector)

                return Contribution(source=node,
                                  target=target_node,
                                  contribution=attribution.item(),
                                  vector=contribution_vector)
    def mlp_contribution(self, last_contribution:Contribution, layer_index:int) -> list[Contribution]:

        contribution_direction = last_contribution.vector

        # the previous source is the new target
        token_position = last_contribution.source.token_position
        target_node = last_contribution.source
        # get all the nodes that are in the same token position and layer
        nodes = self.nodes_by_layer_and_token[layer_index][token_position]
        # filter out the attention nodes and the embedding node
        nodes = [node for node in nodes if not isinstance(node, AttentionNode) and not isinstance(node, InputNode)]
        # get the contribution of each node
        contributions = []
        for node in tqdm(nodes,desc="Computing Node contributions",disable=True):
            new_contribution = self.compute_mlp_node_contribution(
                node,
                target_node,
                contribution_direction,
            )
            contributions.append(new_contribution)

        return contributions


    def compute_node_head_contribution(self, node:Node,
                                      target_node:Node,
                                      vector:Tensor) -> Contribution:
            assert isinstance(node, AttentionNode)
            head = node.head
            token_position = node.token_position
            layer_index = node.layer_index

            #how much is the node contributing to the upstream node

            dot_product = torch.dot(vector,node.output_vector)
            attribution = dot_product

            # attention feature vector
            contribution_vector = node.input_vector@vector
            ln_constant = self.layer_norm_constant(contribution_vector,
                                                   layer_index,
                                                   token_position,
                                                   is_ln2=False)
            contribution_vector = contribution_vector * ln_constant
            new_contribution = AttentionContribution(source=node,
                                            target=target_node,
                                            contribution=attribution.item(),
                                            vector=contribution_vector,
                                            head=head)

            return new_contribution


    def attention_contribution(self,last_contribution:Contribution,layer_index:int) -> list[Contribution]:

        contribution_direction = last_contribution.vector

        # the previous source is the new target
        token_position = last_contribution.source.token_position
        target_node = last_contribution.source

        # get all the nodes that are in the same layer position and any token position
        # before the target node
        nodes_dict = self.nodes_by_layer_and_token[layer_index]
        nodes = []
        # need to include the target node token position
        for tok in range(token_position+1):
            all_nodes = nodes_dict[tok]
            attention_nodes = [node for node in all_nodes if isinstance(node, AttentionNode)]
            nodes.extend(attention_nodes)

        contributions = []

        weighted_attention_head_contribution = self.weighted_attention_head_contribution[layer_index]
        contrib = weighted_attention_head_contribution[0,:,token_position,:]@contribution_direction
        # get the top 10 contributions,
        # TODO: we can potentially remove this part
        _, top_attn_contrib_indices_flattened = torch.topk(contrib.flatten(), k=min([50, len(contrib)]))
        top_attn_contrib_indices = np.array(np.unravel_index(top_attn_contrib_indices_flattened.cpu().numpy(), contrib.shape)).T.tolist()
        for node in tqdm(nodes,desc="Computing Node contributions",disable=True):
            node_head = node.head
            node_token_position = node.token_position
            if [node_head,node_token_position] in top_attn_contrib_indices:
                new_contribution = self.compute_node_head_contribution(node,
                                target_node,
                                contribution_direction)
                contributions.append(new_contribution)
        return  contributions

    @torch.no_grad()
    def flow_once(self,queue:list[Path]=[]):

        # if the queue is empty, get the output node with the highest probability
        # TODO: handle the other output nodes
        if len(queue) == 0:
            output_nodes = [node for node in self.nodes.values() if isinstance(node, OutputNode)]
            highest_probability_node = max(output_nodes, key=lambda x: x.probability)
            contribution = Contribution(source=highest_probability_node,
                                        target=highest_probability_node,
                                        contribution=1,
                                        vector=highest_probability_node.input_vector)
            path = Path(contributions=[contribution])
            print("Doing output node")
        else:
            path = queue[0]
            queue = queue[1:]
            print(f"Doing path: {path.id}, with contribution: {path.total_contribution}")
        self.paths.append(path)
        last_contribution = path.contributions[-1]
        target_node = last_contribution.source
        # compute all the mlp contributions
        max_layer = target_node.layer_index
        all_mlp_contributions = []
        for layer in tqdm(range(max_layer),desc=f"Computing MLP contributions of node {target_node.id}",disable=True):
            contributions = self.mlp_contribution(last_contribution,layer)
            all_mlp_contributions.extend(contributions)

        all_attn_contributions = []
        if isinstance(last_contribution.source, IntermediateNode):
            max_layer += 1
        for layer in tqdm(range(max_layer),desc=f"Computing attention contributions of node {target_node.id}",disable=True):
            contributions = self.attention_contribution(last_contribution,layer)
            all_attn_contributions.extend(contributions)

        # embedding contribution
        embed_node = [node for node in self.nodes_by_layer_and_token[0][target_node.token_position] if isinstance(node, InputNode)][0]
        contribution = torch.dot(last_contribution.vector,embed_node.output_vector).item()
        vector = torch.zeros_like(embed_node.output_vector)
        embedding_contribution = Contribution(source=embed_node,
                                              target=target_node,
                                              contribution=contribution,
                                              vector=vector)

        all_contributions = all_mlp_contributions + all_attn_contributions + [embedding_contribution]
        all_contributions.sort(key=lambda x: x.contribution, reverse=True)

        # Make new paths using the last path
        new_paths = []
        for n_path in range(0,len(all_contributions)):
            new_contributions = path.contributions+[all_contributions[n_path]]
            new_path = Path(contributions=new_contributions)
            # if path ends with input node, error node or skip node, it is finished and we don't want to add it to the queue
            if isinstance(new_path.contributions[-1].source, InputNode) or isinstance(new_path.contributions[-1].source, ErrorNode) or isinstance(new_path.contributions[-1].source, SkipNode):
                self.paths.append(new_path)
                continue
            new_paths.append(new_path)
            if len(new_paths) > 5:
                break
        # sometimes sort by total contribution
        # if random.random() < 0.1:
        #     new_paths.sort(key=lambda x: x.total_contribution, reverse=True)
        # else:
        #     # the other times sort by the past contribution
        #     new_paths.sort(key=lambda x: abs(x.contributions[-1].contribution), reverse=True)
        queue.extend(new_paths)
        queue.sort(key=lambda x: abs(x.contributions[-1].contribution), reverse=True)
        # sort the queue with the total contribution of each path
        return queue

    def flow(self,num_iterations:int=100000):
        queue = []
        for _ in range(num_iterations):
            print(f"Iteration {_}")

            queue = self.flow_once(queue)
            print(f"Queue has {len(queue)} paths")
            top_5_queue = queue[:20]
            print("Top 5 queue:")
            for path in top_5_queue:
                print(f"Path: {path.id} with contribution: {path.total_contribution}")
            # every 10 iterations, save the graph
            if _ % 10 == 0:
                self.save_graph()
