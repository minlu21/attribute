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
    contribution: Tensor
    vector: Tensor


@dataclass
class AttentionContribution(Contribution):
    head: int

@dataclass
class OutputNode(Node):
    token_str: str
    probability: float
    logit: float
    #input_nodes: list[Node]
    input_vector: Tensor

@dataclass
class InputNode(Node):
    token_str: str
    #output_nodes: list[Node]
    output_vector: Tensor
    list_contributions: list[Contribution]
@dataclass
class IntermediateNode(Node):
    feature_index: int
    #input_nodes: list[Node]
    #output_nodes: list[Node]
    activation: Tensor
    input_vector: Tensor
    output_vector: Tensor
    list_contributions: list[Contribution]
    @property
    def total_contribution(self):
        return sum([contribution.contribution.item() for contribution in self.list_contributions])

@dataclass
class SkipNode(Node):
    #output_nodes: list[Node]
    list_contributions: list[Contribution]
    output_vector: Tensor

@dataclass
class ErrorNode(Node):
    #output_nodes: list[Node]
    list_contributions: list[Contribution]
    output_vector: Tensor

@dataclass
class AttentionNode(Node):
    head: int
    source_token_position: int
    list_contributions: list[Contribution]
    input_vector: Tensor
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
        # I want to have a 2D array, where the first dimension is the layer,
        # the second is the token position and then each of these can have a list of nodes
        self.nodes_by_layer_and_token : dict[int,dict[int,list[Node]]] = {}
        
    def save_graph(self):
        for node in self.nodes.values():

            if not isinstance(node, OutputNode):
                node_edges = {}
                for contribution in node.list_contributions:
                    if contribution.target.id not in node_edges:
                        node_edges[contribution.target.id] = 0
                    node_edges[contribution.target.id] += contribution.contribution.item()
                for target_id, contributions in node_edges.items():
                    edge_id = f"{node.id} -> {target_id}"
                    edge = Edge(id=edge_id, source=node, target=self.nodes[target_id], weight=contributions)
                    self.edges[edge_id] = edge
                    #print(f"{node.id} -> {target_id} : {contributions}")
        dict_repr = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
        with open("attribution_graph.json", "w") as f:
            json.dump(dict_repr, f)
                    

    def initialize_graph(self):
        num_layers = len(self.transcoders) 
        seq_len = self.input_ids.shape[0]
        self.nodes_by_layer_and_token = {layer: {token: [] for token in range(seq_len)} for layer in range(num_layers)}

        # Start by creating the input nodes
        for i in range(self.input_ids.shape[0]):
            embedding = self.model.model.embed_tokens.weight[self.input_ids[i]]
            input_node = InputNode(id=f"input_{i}",
                                   token_position=i,
                                   token_str=self.tokenizer.decode([self.input_ids[i]]),
                                   list_contributions=[],
                                   output_vector=embedding.to(dtype=torch.bfloat16),
                                   layer_index=0)
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
                                                        #input_nodes=[],
                                                        #output_nodes=[],
                                                        activation=act,
                                                        input_vector=encoder_direction.to(dtype=torch.bfloat16),
                                                        output_vector=decoder_direction.to(dtype=torch.bfloat16),
                                                        list_contributions=[])
                    self.nodes[intermediate_node.id] = intermediate_node
                    self.nodes_by_layer_and_token[layer_index][token_position].append(intermediate_node)
                # Create the error and skip nodes
                error = self.errors[key][token_position]
                skip = self.skip_connections[key][token_position]
                error_node = ErrorNode(id=f"error_{token_position}_{layer_index}",
                                    layer_index=layer_index,
                                    token_position=token_position,
                                    list_contributions=[],
                                    output_vector=error.to(dtype=torch.bfloat16))
                self.nodes_by_layer_and_token[layer_index][token_position].append(error_node)
                self.nodes[error_node.id] = error_node
                skip_node = SkipNode(id=f"skip_{token_position}_{layer_index}",
                                    layer_index=layer_index,
                                    token_position=token_position,
                                    list_contributions=[],
                                    output_vector=skip.to(dtype=torch.bfloat16))
                self.nodes[skip_node.id] = skip_node
                self.nodes_by_layer_and_token[layer_index][token_position].append(skip_node)
        # Create the attention nodes
        for layer_index in range(len(self.attention_patterns)):
            for head in range(len(self.attention_patterns[layer_index][0])):
                for source_token_position in range(seq_len):
                    for target_token_position in range(source_token_position+1):


                        attention_node = AttentionNode(id=f"attention_{layer_index}_{head}_{source_token_position}_{target_token_position}",
                                                      layer_index=layer_index,
                                                      head=head,
                                                      token_position=target_token_position,
                                                      source_token_position=source_token_position,
                                                      list_contributions=[],
                                                      input_vector=self.OVs[layer_index][head]*self.attention_patterns[layer_index][0,head,target_token_position,source_token_position],
                                                      output_vector=self.OVs[layer_index][head])
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
                                     #input_nodes=[],
                                     input_vector=gradient[0,-1].to(dtype=torch.bfloat16),
                                     layer_index=len(self.transcoders))
            print(f"Output node {i}: with probability {top_10_probabilities[i]}, string {self.tokenizer.decode([top_10_indices[i]])}")
            self.nodes[output_node.id] = output_node
            total_probability += top_10_probabilities[i]
            if total_probability > 0.95:
                break
        # Total number of nodes
        print(f"Total number of nodes: {len(self.nodes)}")
        # Count each type of node
        input_nodes = [node for node in self.nodes if isinstance(node, InputNode)]
        intermediate_nodes = [node for node in self.nodes if isinstance(node, IntermediateNode)]
        output_nodes = [node for node in self.nodes if isinstance(node, OutputNode)]
        print(f"Number of input nodes: {len(input_nodes)}")
        print(f"Number of intermediate nodes: {len(intermediate_nodes)}")
        print(f"Number of output nodes: {len(output_nodes)}")
        self.compute_weighted_attention_head_contribution()
        
    def compute_weighted_attention_head_contribution(self):
        weighted_attention_head_contribution = []
        # We can store attention contribution instead of all the patterns and values
        OVs = []
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
 

    def layer_norm_constant(self,vector,layer_index,token_position,is_ln2=False):
        if is_ln2:
            ln = self.second_ln[f"model.layers.{layer_index}.post_attention_layernorm"][0,token_position]
            pre_ln = self.pre_second_ln[f"model.layers.{layer_index}.post_attention_layernorm"][0,token_position]
        else:
            ln = self.first_ln[f"model.layers.{layer_index}"][0,token_position]
            pre_ln = self.pre_first_ln[f"model.layers.{layer_index}"][0,token_position]
        ln_constant = torch.dot(vector,pre_ln)
        if ln_constant == 0:
            return ln_constant * vector
        else:
            return torch.norm(ln)/torch.norm(pre_ln)
        
    def compute_mlp_node_contribution(self, node, target_node, vector, layer_index, token_position):
                if isinstance(node, IntermediateNode) :
                    activation = node.activation
                    input_norm = 1#/self.input_norm[f"model.layers.{layer_index}.mlp"][token_position][0]
                    
                elif isinstance(node, SkipNode):
                    activation = 1
                    input_norm = 1#/self.input_norm[f"model.layers.{layer_index}.mlp"][token_position][0]
                    
                else:
                    activation = 1
                    input_norm = 1
                    
                dot_product = torch.dot(vector, node.output_vector)
                attribution = activation * dot_product
                # apply layer norm
                ln_constant = self.layer_norm_constant(vector, layer_index, token_position, is_ln2=True)
                if isinstance(node,IntermediateNode):
                    contribution_vector = dot_product * node.input_vector * input_norm * ln_constant 
               
                else:
                    contribution_vector = torch.zeros_like(node.output_vector)
                #print(f"Output node {target_node.id} actribution: {attribution}")
                
                return Contribution(source=node,
                                  target=target_node, 
                                  contribution=attribution,
                                  vector=contribution_vector), attribution
    def mlp_contribution(self, target_node, layer_index):
        
        token_position = target_node.token_position
        # get all the nodes that are in the same token position and layer
        nodes = self.nodes_by_layer_and_token[layer_index][token_position]

        # get the contribution of each node
 
        contributions = []
        for node in tqdm(nodes,desc="Computing Node contributions",disable=True):
            if isinstance(target_node, IntermediateNode):
                sorted_contributions = sorted(target_node.list_contributions, key=lambda x: x.contribution, reverse=True)
                top_5_contributions = sorted_contributions[:5]
                for contribution in top_5_contributions:
                    vector = contribution.vector
                    new_contribution, attribution = self.compute_mlp_node_contribution(node, target_node, vector, layer_index, token_position)
                    
                    contributions.append(new_contribution)
            else:
                vector = target_node.input_vector
                new_contribution, attribution = self.compute_mlp_node_contribution(node, target_node, vector, layer_index, token_position)
                contributions.append(new_contribution)
                #print(f"Node: {node.id} with contribution: {new_contribution.contribution}")
                
        return contributions


    def compute_node_head_contribution(self, node, target_node, vector, layer_index, token_position,head,contrib):
            if isinstance(node, IntermediateNode) :
                activation = node.activation.item()
                norm = 1#/self.input_norm[f"model.layers.{layer_index}.mlp"][token_position][0].item()
            elif isinstance(node, SkipNode):
                activation = 1
                norm = 1#/self.input_norm[f"model.layers.{layer_index}.mlp"][token_position][0].item()
            else:
                activation = 1
                norm = 1
            
            # attention feature vector
            contribution_vector = self.OVs[layer_index][head] @ vector
            contribution_vector = self.attention_patterns[layer_index][0,head,token_position,node.token_position] * contribution_vector
            ln_constant = self.layer_norm_constant(contribution_vector,layer_index,token_position,is_ln2=False)
            contribution_vector = contribution_vector * ln_constant#* contrib[head,node.token_position]
            
            #how much is the latent contributing to that feature vector
            
            dot_product = torch.dot(contribution_vector,node.output_vector) * norm 
            
            attribution = activation * dot_product 
            if head == 5:
                print(attribution.item(),activation
                    ,dot_product.item(),
                    self.attention_patterns[layer_index][0,head,token_position,node.token_position].item(),
                    ln_constant.item()
                    ,head,node.token_position,token_position)
                
            if isinstance(node, IntermediateNode):
                contribution_vector = dot_product * node.input_vector
                l2_constant = self.layer_norm_constant(contribution_vector,layer_index,token_position,is_ln2=True)
                contribution_vector = contribution_vector  * l2_constant 
            else:
                contribution_vector = torch.zeros_like(node.output_vector)
            new_contribution = AttentionContribution(source=node,
                                            target=target_node,
                                            contribution=attribution,
                                            vector=contribution_vector,
                                            head=head)
            
            return new_contribution, attribution


    def attention_contribution(self,target_node,layer_index):
        
        token_position = target_node.token_position

        # get all the nodes that are in the same layer position and any token position
        # before the target node
        nodes_dict = self.nodes_by_layer_and_token[layer_index]
        nodes = []
        # need to include the target node token position
        for token_position in range(token_position+1):
            nodes.extend(nodes_dict[token_position])
        
        contributions = []
      
        if isinstance(target_node, IntermediateNode):
            sorted_contributions = sorted(target_node.list_contributions, key=lambda x: x.contribution, reverse=True)
            top_5_contributions = sorted_contributions[:5]
            for contribution in top_5_contributions:
                current_contribution = contribution.contribution
                vector = contribution.vector
                weighted_attention_head_contribution = self.weighted_attention_head_contribution[layer_index]
                contrib = weighted_attention_head_contribution[0,:,token_position,:]@vector
                # get the top 10 contributions
                top_attn_contribs_flattened, top_attn_contrib_indices_flattened = torch.topk(contrib.flatten(), k=min([50, len(contrib)]))
                top_attn_contrib_indices = np.array(np.unravel_index(top_attn_contrib_indices_flattened.cpu().numpy(), contrib.shape)).T.tolist()
                for node in tqdm(nodes,desc="Computing Node contributions",disable=True):
                        for head in range(len(self.attention_patterns[layer_index][0])):
                            if [head,node.token_position] in top_attn_contrib_indices:
                                new_contribution, attribution = self.compute_node_head_contribution(node,
                                                target_node,
                                                vector,
                                                layer_index,
                                                token_position,head,contrib)
                                #new_contribution.contribution = new_contribution.contribution * current_contribution
                                contributions.append(new_contribution)
        else:
            vector = target_node.input_vector
            weighted_attention_head_contribution = self.weighted_attention_head_contribution[layer_index]
            contrib = weighted_attention_head_contribution[0,:,token_position,:]@vector
                
            top_attn_contribs_flattened, top_attn_contrib_indices_flattened = torch.topk(contrib.flatten(), k=min([50, len(contrib)]))
            top_attn_contrib_indices = np.array(np.unravel_index(top_attn_contrib_indices_flattened.cpu().numpy(), contrib.shape)).T.tolist()
                
            for node in tqdm(nodes,desc="Computing Node contributions",disable=True):
                        for head in range(len(self.attention_patterns[layer_index][0])):
                            if [head,node.token_position] in top_attn_contrib_indices:
                                #print(f"Node: {node.id} with head: {head} and token position: {node.token_position}")
                                #print(top_attn_contrib_indices)
                                new_contribution, attribution = self.compute_node_head_contribution(node,
                                                target_node,
                                                vector,
                                                layer_index,
                                                token_position,head,contrib)
                                #print("Node: ",node.id,"Contribution: ",new_contribution.contribution)
                                contributions.append(new_contribution)
        return  contributions#nodes_altered, total_contribution, embedding_contribution, feature_contribution, skip_contribution, error_contribution
        
    @torch.no_grad()
    def flow_once(self,queue:list[IntermediateNode]=[]):
        
        # if the queue is empty, get the output node with the highest probability
        if len(queue) == 0:
            output_nodes = [node for node in self.nodes.values() if isinstance(node, OutputNode)]
            node_to_flow = max(output_nodes, key=lambda x: x.probability)
            current_contribution = 1
        else:
            node_to_flow = queue[0]
            queue = queue[1:]
            current_contribution = node_to_flow.total_contribution
        print(f"Node to flow: {node_to_flow.id}")
        
        # compute all the mlp contributions
        max_layer_mlp = node_to_flow.layer_index if isinstance(node_to_flow, OutputNode) else node_to_flow.layer_index-1
       
        all_mlp_contributions = []
        for layer in tqdm(range(max_layer_mlp),desc=f"Computing MLP contributions of node {node_to_flow.id}",disable=True):
            contributions = self.mlp_contribution(node_to_flow,layer)
            all_mlp_contributions.extend(contributions)
           
        
        # get the top 10 contributions
        all_mlp_contributions.sort(key=lambda x: x.contribution, reverse=True)
        top_10_contributions = all_mlp_contributions[:100]
        for contribution in top_10_contributions:
            
            node = contribution.source
            node.list_contributions.append(contribution)
            print(f"Node: {node.id} with contribution: {contribution.contribution}")
            if isinstance(node, IntermediateNode):
                if node not in queue:
                    # in the case of attention, multiple heads can have the same node
                    queue.append(node)
        # get the bottom 10 contributions
        bottom_10_contributions = all_mlp_contributions[-20:]
        for contribution in bottom_10_contributions:
            node = contribution.source
            node.list_contributions.append(contribution)
                
            if isinstance(node, IntermediateNode):
                if node not in queue:
                    # in the case of attention, multiple heads can have the same node
                    queue.append(node)
        # total_mlp_contribution = sum([contribution.contribution if isinstance(contribution, Contribution) else 0 for contribution in all_mlp_contributions])
        # embedding_mlp_contribution = sum([contribution.contribution if isinstance(contribution.source, InputNode) else 0 for contribution in all_mlp_contributions])
        # feature_mlp_contribution = sum([contribution.contribution if isinstance(contribution.source, IntermediateNode) else 0 for contribution in all_mlp_contributions])
        # skip_mlp_contribution = sum([contribution.contribution if isinstance(contribution.source, SkipNode) else 0 for contribution in all_mlp_contributions])
        # error_mlp_contribution = sum([contribution.contribution if isinstance(contribution.source, ErrorNode) else 0 for contribution in all_mlp_contributions])

        # compute the attention contribution
        max_layer_attn = node_to_flow.layer_index if isinstance(node_to_flow, OutputNode) else node_to_flow.layer_index-1
       
        all_attn_contributions = []
        for layer in tqdm(range(max_layer_attn),desc=f"Computing attention contributions of node {node_to_flow.id}",disable=True):
            contributions = self.attention_contribution(node_to_flow,layer)
            
            all_attn_contributions.extend(contributions)
        print(f"Total attention contributions: {len(all_attn_contributions)}")
        all_attn_contributions.sort(key=lambda x: x.contribution, reverse=True)
        top_10_attn_contributions = all_attn_contributions[:50]
        for contribution in top_10_attn_contributions:
           
            node = contribution.source
            
            node.list_contributions.append(contribution)
            print(f"Node: {node.id} with contribution: {contribution.contribution}")
            if isinstance(node, IntermediateNode):
                if node not in queue:
                    queue.append(node)
        bottom_10_attn_contributions = all_attn_contributions[-10:]
        for contribution in bottom_10_attn_contributions:
            node = contribution.source
            node.list_contributions.append(contribution)
                
            if isinstance(node, IntermediateNode):
                if node not in queue:
                    queue.append(node)
        queue.sort(key=lambda x: x.total_contribution, reverse=True)

        # return the queue
        return queue

    def flow(self,num_iterations:int=1):
        queue = []
        total_contribution = 0
        for _ in range(num_iterations):
            print(f"Iteration {_}")
            
            queue = self.flow_once(queue)
            print(f"Doing contribution: {queue[0].total_contribution}")
            print(f"Queue has {len(queue)} nodes")
            contribution_of_queue = sum([node.total_contribution for node in queue])
            print(f"Total contribution: {total_contribution}")
            print(f"Contribution of queue: {contribution_of_queue}")
            total_contribution += queue[0].total_contribution
            top_5_queue = queue[:5]
            print(f"Top 5 queue:")
            for node in top_5_queue:
                print(f"Node: {node.id} with contribution: {node.total_contribution}")
        
        self.save_graph()
            