"""
Example usage of the reasoning head discovery and evaluation framework.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set default cache directory
DEFAULT_CACHE_DIR = "/cluster/scratch/yongyu/cache"

from reasoning_heads import (
    ReasoningHeadDiscovery,
    ReasoningHeadEvaluator,
    generate_evaluation_report
)


def example_discovery(cache_dir: str = DEFAULT_CACHE_DIR):
    """Example: Discover reasoning heads."""
    print("="*60)
    print("Example: Discovering Reasoning Heads")
    print("="*60)
    
    # Set cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Loading model: {model_name}")
    print(f"Using cache directory: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    ).eval()
    
    # Initialize discovery
    discovery = ReasoningHeadDiscovery(
        model=model,
        tokenizer=tokenizer,
        backward_chaining_dir="../../backward-chaining-circuits",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_method="ablation",
        cache_dir=cache_dir
    )
    
    # Discover heads
    print("\nDiscovering reasoning heads...")
    discovered_heads = discovery.discover_heads(
        n_examples_per_subtask=10,  # Use fewer examples for quick demo
        top_k=5  # Top 5 heads per subtask
    )
    
    # Save results
    discovery.save_discovered_heads(discovered_heads, "example_heads.json")
    print(f"\nDiscovered {len(discovered_heads)} reasoning heads")
    print(f"Saved to example_heads.json")
    
    # Show some examples
    print("\nExample discovered heads:")
    for head in discovered_heads[:5]:
        print(f"  Layer {head.layer}, Head {head.head}: "
              f"score={head.score:.4f}, subtask={head.subtask}")


def example_evaluation(cache_dir: str = DEFAULT_CACHE_DIR):
    """Example: Evaluate discovered heads on benchmarks."""
    print("\n" + "="*60)
    print("Example: Evaluating on Benchmarks")
    print("="*60)
    
    # Set cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model (same as discovery)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Using cache directory: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    ).eval()
    
    # Initialize discovery and evaluator
    discovery = ReasoningHeadDiscovery(
        model=model,
        tokenizer=tokenizer,
        backward_chaining_dir="../../backward-chaining-circuits",
        cache_dir=cache_dir
    )
    
    # Load discovered heads
    if os.path.exists("example_heads.json"):
        discovered_heads = discovery.load_discovered_heads("example_heads.json")
        print(f"Loaded {len(discovered_heads)} discovered heads")
    else:
        print("No discovered heads file found. Run discovery first.")
        return
    
    # Initialize evaluator
    evaluator = ReasoningHeadEvaluator(discovery)
    
    # Define benchmark configs
    testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_configs = {
        "hotpotqa": {
            "main": os.path.join(testing_dir, "configs", "hotpotqa_model_config.yaml"),
            "baseline": os.path.join(testing_dir, "configs", "hotpotqa_baseline_config.yaml")
        },
        # Add other benchmarks as needed
    }
    
    # Evaluate
    print("\nEvaluating benchmarks...")
    results = evaluator.evaluate_all_benchmarks(
        discovered_heads=discovered_heads,
        benchmark_configs=benchmark_configs,
        top_k_heads=5  # Mask top 5 heads
    )
    
    # Save results
    evaluator.save_results(results, "example_results.json")
    
    # Generate report
    print("\nGenerating report...")
    report_data = {}
    for benchmark_name, (baseline, masked) in results.items():
        comparison = evaluator.compare_results(baseline, masked)
        report_data[benchmark_name] = {
            "baseline": baseline.to_dict(),
            "masked": masked.to_dict(),
            "comparison": comparison
        }
    
    report = generate_evaluation_report(
        results=report_data,
        discovered_heads=discovered_heads,
        output_file="example_report.md"
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to example_results.json")
    print(f"Report saved to example_report.md")


def example_subtask_analysis(cache_dir: str = DEFAULT_CACHE_DIR):
    """Example: Analyze specific subtask."""
    print("\n" + "="*60)
    print("Example: Subtask Analysis")
    print("="*60)
    
    # Set cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Using cache directory: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    ).eval()
    
    discovery = ReasoningHeadDiscovery(
        model=model,
        tokenizer=tokenizer,
        backward_chaining_dir="../../backward-chaining-circuits",
        cache_dir=cache_dir
    )
    
    # Get heads for specific subtask
    if os.path.exists("example_heads.json"):
        all_heads = discovery.load_discovered_heads("example_heads.json")
        
        # Filter by subtask
        path_finding_heads = discovery.get_heads_for_subtask(all_heads, "path_finding")
        
        print(f"\nFound {len(path_finding_heads)} heads for 'path_finding' subtask:")
        for head in path_finding_heads[:5]:
            print(f"  Layer {head.layer}, Head {head.head}: score={head.score:.4f}")
        
        # Get head list for masking
        head_list = discovery.get_head_list_for_masking(
            path_finding_heads,
            top_k=5
        )
        print(f"\nHead list for masking (top 5): {head_list}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["discovery", "evaluation", "analysis", "all"],
        default="all",
        help="Which step to run"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for models (default: {DEFAULT_CACHE_DIR})"
    )
    
    args = parser.parse_args()
    
    if args.step in ["discovery", "all"]:
        example_discovery(cache_dir=args.cache_dir)
    
    if args.step in ["evaluation", "all"]:
        example_evaluation(cache_dir=args.cache_dir)
    
    if args.step in ["analysis", "all"]:
        example_subtask_analysis(cache_dir=args.cache_dir)

