# Evaluation script for measuring model performance on next token prediction
# Calculates token-level accuracy and KL divergence between predicted and true distributions
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import sys
sys.path.append("..")

import IPython
if ip := IPython.get_ipython():
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

from attribute.caching import TranscodedModel
from attribute.mlp_attribution import AttributionGraph, AttributionConfig
from sparsify.data import chunk_and_tokenize
from datasets import load_dataset
from multiprocessing import cpu_count
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model accuracy and KL divergence on next token prediction')
    parser.add_argument('--model', type=str, default="nev/GELU_4L512W_C4_Code",
                      help='Model name or path')
    parser.add_argument('--transcoder_path', type=str,
                      default="../e2e/checkpoints/gelu-4l-clt/ef128k64",
                      help='Path to transcoder')
    parser.add_argument('--mlp_trim', type=int,
                      default=0,
                      help='Fix MLP L0 to this value')
    parser.add_argument('--output_path', type=str, default="results/accuracy",
                      help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for evaluation')
    parser.add_argument('--max_seq_len', type=int, default=64,
                      help='Maximum sequence length for evaluation')
    parser.add_argument('--max_steps', type=int, default=128,
                      help='Maximum number of steps for evaluation')
    parser.add_argument('--pre_ln_hook', action='store_true',
                      help='Use pre-layer-norm hook')
    parser.add_argument('--post_ln_hook', action='store_true',
                      help='Use post-layer-norm hook')
    parser.add_argument('--calculate_completeness', action='store_true',
                      help='Calculate completeness score (requires transcoder)')
    parser.add_argument('--calculate_replacement', action='store_true',
                      help='Calculate replacement score (requires transcoder)')
    parser.add_argument('--remove_prefix', type=int, default=0,
                      help='Number of tokens to remove from the beginning (similar to serve.py)')
    return parser.parse_args()

def calculate_attribution_scores(model, input_ids, transcoder_path=None, pre_ln_hook=False, post_ln_hook=False, remove_prefix=0):
    """
    Calculate both completeness and replacement scores using the same attribution graph.
    
    Completeness score: measures the fraction of input edges (weighted by the target node's logit influence score) 
    that come from feature or embedding nodes rather than error nodes.
    
    Replacement score: measures the fraction of end-to-end graph paths (weighted by strength) that proceed 
    from embedding nodes to logit nodes via feature nodes (rather than error nodes).
    
    Args:
        model: The model to evaluate
        input_ids: Input token IDs
        transcoder_path: Path to transcoder (optional)
        pre_ln_hook: Whether to use pre-layer-norm hook
        post_ln_hook: Whether to use post-layer-norm hook
        remove_prefix: Number of tokens to remove from the beginning (similar to serve.py)
    
    Returns:
        dict: Dictionary containing 'completeness_score' and 'replacement_score' (both between 0 and 1)
    """
    try:
        # Create a simple attribution config for the calculation
        config = AttributionConfig(
            name="attribution_eval",
            scan="eval",
            flow_steps=500,  # Reduced for efficiency and stability
            batch_size=16,   # Reduced batch size for stability
            softmax_grad_type="mean",
            node_cum_threshold=0.8,
            edge_cum_threshold=0.98,
            top_k_edges=16,  # Reduced for stability
            influence_anthropic=True,
            influence_gpu=True,
            pre_filter_threshold=1e-5,
            keep_all_error_nodes=True,
            keep_all_input_nodes=True,
            use_all_targets=False,
            filter_high_freq_early=0.01,
            use_logit_bias=False,
            use_self_explanation=False
        )
        
        # Get model outputs with transcoder if available
        if transcoder_path:
            # Use the same pattern as serve.py - get transcoded outputs properly
            # Convert input_ids back to text prompt for the model
            prompt = model.tokenizer.decode(input_ids[0])
            transcoded_outputs = model([prompt] * config.batch_size)
            # Remove prefix if needed (similar to serve.py)
            transcoded_outputs.remove_prefix(remove_prefix)  # Adjust this if needed
            out = transcoded_outputs
        else:
            # For models without transcoder, we need to compute with gradients
            out = model.model(input_ids)
        
        # Create attribution graph
        graph = AttributionGraph(model, out, config)
        
        # Run attribution flow to build the graph with error handling
        try:
            graph.flow()
        except RuntimeError as e:
            if "CUDA error" in str(e) or "index out of bounds" in str(e):
                print(f"CUDA error during attribution flow, trying with reduced flow steps: {e}")
                # Try with even fewer flow steps
                config.flow_steps = 100
                config.batch_size = 8
                config.top_k_edges = 8
                graph = AttributionGraph(model, out, config)
                graph.flow()
            else:
                raise e
        
        # Calculate adjacency matrix and influence
        adj_matrix, dedup_node_names, activation_sources = graph.adjacency_matrix(absolute=True, normalize=False)
        dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}
        
        n_initial = len(dedup_node_names)
        
        # Create masks for different node types
        error_mask = np.zeros((n_initial,))
        input_mask = np.zeros((n_initial,))
        feature_mask = np.zeros((n_initial,))
        output_mask = np.zeros((n_initial,))
        
        for i, node in enumerate(dedup_node_names):
            node_obj = graph.nodes[node]
            if hasattr(node_obj, 'node_type'):
                if node_obj.node_type == "ErrorNode":
                    error_mask[i] = 1
                elif node_obj.node_type == "InputNode":
                    input_mask[i] = 1
                elif node_obj.node_type == "IntermediateNode":
                    feature_mask[i] = 1
                elif node_obj.node_type == "OutputNode":
                    output_mask[i] = 1
        
        # Create influence sources - identify output nodes and their probabilities
        influence_sources = np.zeros((n_initial,))
        for index, node in enumerate(dedup_node_names):
            node_obj = graph.nodes[node]
            if hasattr(node_obj, 'node_type') and node_obj.node_type == "OutputNode":
                influence_sources[index] = node_obj.probability
        
        # Calculate influence matrix
        influence, adj_matrix = graph.find_influence(adj_matrix)
        
        # Calculate completeness score: 1 - (influence_sources @ adj_matrix) @ error_mask
        # This measures the fraction of influence that comes from non-error nodes
        completeness_score = 1 - (influence_sources @ adj_matrix) @ error_mask
        
        # Calculate replacement score: 1 - (influence_sources @ influence) @ error_mask
        # This measures the fraction of end-to-end paths that don't go through error nodes
        replacement_score = 1 - (influence_sources @ influence) @ error_mask
        
        return {
            'completeness_score': float(completeness_score),
            'replacement_score': float(replacement_score)
        }
        
    except Exception as e:
        print(f"Error in calculate_attribution_scores: {e}")
        import traceback
        traceback.print_exc()
        # Return default values if calculation fails
        return {
            'completeness_score': 0.0,
            'replacement_score': 0.0
        }

def main():
    args = parse_args()

    # Initialize model with specified transcoder for feature attribution
    model = TranscodedModel(
        args.model,
        transcoder_path=args.transcoder_path if args.mlp_trim == 0 else None,
        pre_ln_hook=args.pre_ln_hook,
        post_ln_hook=args.post_ln_hook,
    )

    # Load and preprocess dataset into fixed-length chunks for efficient batching
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    dataset = chunk_and_tokenize(
        dataset,
        model.tokenizer,
        max_seq_len=args.max_seq_len,
        num_proc=cpu_count() // 2,
        text_key="text",
        return_overflowed_tokens=True,
    )

    # Setup data loading with multiprocessing
    logger.remove()  # Remove default logger to avoid clutter during evaluation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=cpu_count() // 2
    )

    # Track metrics across batches
    accuracies = []  # Token-level accuracy
    kl_divs = []    # KL divergence between predicted and true distributions
    completeness_scores = []  # Completeness scores
    replacement_scores = []   # Replacement scores

    def mlp_trimmer(module, input, output):
        values, indices = torch.topk(output, args.mlp_trim, dim=-1)
        output = torch.zeros_like(output)
        output.scatter_(dim=-1, index=indices, src=values)
        return output

    # Evaluate on full dataset
    try:
        if args.max_steps is not None:
            dataloader = itertools.islice(dataloader, args.max_steps)
        bar = tqdm(dataloader, desc="Evaluating", total=args.max_steps)
        for batch_idx, batch in enumerate(bar):
            input_ids = batch["input_ids"].to(model.device)
            with torch.no_grad():
                out_base = model.model(input_ids)
                base_logits = out_base.logits
                if args.mlp_trim > 0:
                    for layer_idx in range(model.num_layers):
                        mlp = model.model.get_submodule(f"{model.layer_prefix}.{layer_idx}.mlp")
                        in_proj = getattr(mlp, model.mlp_in_proj_name)
                        in_proj.register_forward_hook(mlp_trimmer)
                    out = model.model(input_ids)
                    for m in model.model.modules():
                        m._forward_hooks = OrderedDict()
                else:
                    out = model(input_ids, no_error="after_first")
                logits = out.logits

                # Calculate token-level accuracy
                pred_tokens = logits.argmax(dim=-1)
                base_tokens = base_logits.argmax(dim=-1)
                accuracy = (pred_tokens == base_tokens).float().mean().item()
                accuracies.append(accuracy)

                # Calculate KL divergence between predicted and true distributions
                log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
                base_log_probs = F.log_softmax(base_logits, dim=-1)  # [batch, seq_len, vocab_size]

                # Calculate KL divergence for this batch
                kl_div = F.kl_div(
                    log_probs.view(-1, logits.size(-1)),
                    base_log_probs.view(-1, logits.size(-1)),
                    reduction='batchmean',
                    log_target=True
                ).item()
                kl_divs.append(kl_div)

                # Calculate attribution scores if requested (only on first batch to avoid excessive computation)
                if (args.calculate_completeness or args.calculate_replacement) and batch_idx == 0:
                    print(f"Starting attribution score calculation...")
                    print(f"calculate_completeness: {args.calculate_completeness}")
                    print(f"calculate_replacement: {args.calculate_replacement}")
                    
                    # Temporarily enable gradients for attribution calculation
                    with torch.enable_grad():
                        try:
                            print(f"Calling calculate_attribution_scores...")
                            attribution_scores = calculate_attribution_scores(
                                model, 
                                input_ids[0:1],  # Use first example only for efficiency
                                transcoder_path=args.transcoder_path if args.mlp_trim == 0 else None,
                                pre_ln_hook=args.pre_ln_hook,
                                post_ln_hook=args.post_ln_hook,
                                remove_prefix=args.remove_prefix
                            )
                            print(f"Attribution scores calculated: {attribution_scores}")
                            
                            if args.calculate_completeness:
                                completeness_score = attribution_scores['completeness_score']
                                completeness_scores.append(completeness_score)
                                print(f"Completeness score: {completeness_score:.4f}")
                            
                            if args.calculate_replacement:
                                replacement_score = attribution_scores['replacement_score']
                                replacement_scores.append(replacement_score)
                                print(f"Replacement score: {replacement_score:.4f}")
                                
                        except Exception as e:
                            print(f"Failed to calculate attribution scores: {e}")
                            import traceback
                            traceback.print_exc()
                            # Add fallback values
                            if args.calculate_completeness:
                                completeness_scores.append(0.0)
                            if args.calculate_replacement:
                                replacement_scores.append(0.0)

                # Update progress bar with current and running average metrics
                postfix = {
                    "Avg Acc": f"{np.mean(accuracies):.4f}",
                    "Avg KL": f"{np.mean(kl_divs):.4f}"
                }
                if completeness_scores and len(completeness_scores) > 0:
                    postfix["Completeness"] = f"{completeness_scores[0]:.4f}"
                if replacement_scores and len(replacement_scores) > 0:
                    postfix["Replacement"] = f"{replacement_scores[0]:.4f}"
                bar.set_postfix(postfix)
    except KeyboardInterrupt:
        pass

    # Calculate final statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_kl = np.mean(kl_divs)
    std_kl = np.std(kl_divs)
    
    # Add attribution scores to results if calculated
    results_extra = {}
    print(f"Final completeness_scores: {completeness_scores}")
    print(f"Final replacement_scores: {replacement_scores}")
    if completeness_scores and len(completeness_scores) > 0:
        results_extra["completeness_score"] = completeness_scores[0]
        print(f"Adding completeness_score to results: {completeness_scores[0]}")
    if replacement_scores and len(replacement_scores) > 0:
        results_extra["replacement_score"] = replacement_scores[0]
        print(f"Adding replacement_score to results: {replacement_scores[0]}")
    print(f"results_extra: {results_extra}")

    # Prepare comprehensive results dictionary
    results = {
        "model": args.model,
        **(
            {"transcoder_path": args.transcoder_path}
            if args.mlp_trim == 0
            else {"mlp_trim": args.mlp_trim}
        ),
        "timestamp": datetime.now().isoformat(),
        "accuracy": {
            "mean": mean_accuracy,
            "std": std_accuracy,
        },
        "kl_divergence": {
            "mean": mean_kl,
            "std": std_kl,
        },
        "metadata": {
            "dataset": "NeelNanda/pile-10k",
            "batch_size": args.batch_size,
            "num_batches": len(accuracies),
            "total_tokens": len(accuracies) * args.batch_size * (args.max_seq_len - 1),  # -1 for next token prediction
        },
        **results_extra
    }
    
    print(f"Final results dictionary keys: {list(results.keys())}")
    if "completeness_score" in results:
        print(f"completeness_score in results: {results['completeness_score']}")
    if "replacement_score" in results:
        print(f"replacement_score in results: {results['replacement_score']}")

    # Save results with timestamp for tracking experiments
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    escaped_model_name = args.model.replace("/", "_").replace(":", "_")
    escaped_transcoder_name = args.transcoder_path.replace("/", "_").replace(":", "_")
    if args.mlp_trim > 0:
        escaped_transcoder_name = f"mlp_trim_{args.mlp_trim}"
    escaped_model_transcoder_name = f"{escaped_model_name}-{escaped_transcoder_name}"
    output_file = output_dir / f"eval_results_{escaped_model_transcoder_name}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Token Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"KL Divergence: {mean_kl:.4f} ± {std_kl:.4f}")
    if completeness_scores and len(completeness_scores) > 0:
        print(f"Completeness Score: {completeness_scores[0]:.4f}")
    if replacement_scores and len(replacement_scores) > 0:
        print(f"Replacement Score: {replacement_scores[0]:.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
