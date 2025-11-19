"""
Standalone script to discover reasoning heads from CognitiveMirrors dataset.

This script:
1. Loads examples from cognitive_mirrors_logical_reasoning.json
2. For each attention head:
   - Runs baseline model (no masking)
   - Runs with head masked (ablated)
   - Compares BLEU scores (baseline should be better)
   - Score = baseline_bleu - ablated_bleu
3. Selects heads with largest positive scores (most important)
4. Saves in DeCoReEntropy format for later use
"""

import os
import sys
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_heads import ReasoningHeadDiscovery

DEFAULT_CACHE_DIR = "/cluster/scratch/yongyu/cache"


def main():
    parser = argparse.ArgumentParser(
        description="Discover reasoning heads from CognitiveMirrors Logical Reasoning dataset"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="ablation",
        choices=["ablation"],
        help="Head scoring method (only ablation supported)"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=20,
        help="Number of examples to use for scoring each head"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K heads to select (heads with largest performance drop when masked)"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum score threshold (baseline - ablated BLEU)"
    )
    parser.add_argument(
        "--output_heads",
        type=str,
        default="discovered_heads.json",
        help="Output file for discovered heads (full format)"
    )
    parser.add_argument(
        "--retrieval_heads_dir",
        type=str,
        default="../retrieval_heads/",
        help="Directory to save DeCoReEntropy format heads"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for models (default: {DEFAULT_CACHE_DIR})"
    )
    
    args = parser.parse_args()
    
    # Set cache directory
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Using cache directory: {args.cache_dir}")
    
    # Load model and tokenizer
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    print(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        from src.factories import get_model
        from src.configs import ModelConfigs, DecoderConfigs, ModelConfig, DecoderConfig
        
        model_configs = ModelConfigs(
            name="LLaMA3-8b-Instruct",
            model_type="instruct",
            configs=ModelConfig(
                model_name_or_path=args.model_name,
                max_seq_len=4096,
                max_new_tokens=100
            )
        )
        
        decoder_configs = DecoderConfigs(
            name="baseline",
            method="baseline",
            configs=ModelConfig()
        )
        
        model_wrapper = get_model(model_configs, decoder_configs)
        model = model_wrapper.model
    except Exception as e:
        print(f"Warning: Could not load model using factory, using direct loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.cache_dir
        ).eval()
    
    # Initialize discovery
    print(f"\n{'='*80}")
    print("INITIALIZING REASONING HEAD DISCOVERY")
    print(f"{'='*80}")
    
    discovery = ReasoningHeadDiscovery(
        model=model,
        tokenizer=tokenizer,
        backward_chaining_dir="../CognitiveMirrors",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_method=args.scoring_method,
        cache_dir=args.cache_dir
    )
    
    # Discover heads
    print(f"\n{'='*80}")
    print("DISCOVERING REASONING HEADS")
    print(f"{'='*80}")
    print(f"Method: Compare baseline vs ablated BLEU scores")
    print(f"Expected: Baseline should have higher BLEU than ablated")
    print(f"Score = baseline_bleu - ablated_bleu (positive = important head)")
    print(f"Selecting top {args.top_k} heads with largest positive scores\n")
    
    discovered_heads = discovery.discover_heads(
        n_examples_per_subtask=args.n_examples,
        top_k=args.top_k,
        single_subtask="logical_reasoning",
        min_score=args.min_score,
        min_confidence=0.0
    )
    
    if len(discovered_heads) == 0:
        print("\nERROR: No reasoning heads discovered!")
        print("This might indicate:")
        print("  1. All heads have negative scores (masking improves performance)")
        print("  2. Baseline and ablated performance are too similar")
        print("  3. Model is not performing well on the task")
        return
    
    # Save in full format
    print(f"\n{'='*80}")
    print("SAVING DISCOVERED HEADS")
    print(f"{'='*80}")
    
    discovery.save_discovered_heads(discovered_heads, args.output_heads)
    print(f"✓ Saved {len(discovered_heads)} heads to {args.output_heads}")
    
    # Save in DeCoReEntropy format
    model_base_name = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
    decore_file = discovery.save_heads_for_decore(
        discovered_heads,
        output_dir=args.retrieval_heads_dir,
        model_name=model_base_name
    )
    
    print(f"\n{'='*80}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Discovered {len(discovered_heads)} reasoning heads")
    print(f"✓ Saved to {args.output_heads} (full format)")
    print(f"✓ Saved to {decore_file} (DeCoReEntropy format)")
    print(f"\nYou can now use these heads in DeCoReEntropy by setting:")
    print(f"  retrieval_heads_dir: '{args.retrieval_heads_dir}'")
    print(f"  num_retrieval_heads: {len(discovered_heads)}")
    print(f"\nTop {min(5, len(discovered_heads))} reasoning heads:")
    sorted_heads = sorted(discovered_heads, key=lambda h: h.score, reverse=True)
    for i, head in enumerate(sorted_heads[:5]):
        print(f"  {i+1}. Layer {head.layer}, Head {head.head}: score={head.score:.4f} "
              f"(baseline_bleu - ablated_bleu)")


if __name__ == "__main__":
    main()

