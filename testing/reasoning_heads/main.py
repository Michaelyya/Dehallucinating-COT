"""
Main CLI script for reasoning head discovery and evaluation.
"""

import argparse
import os
import sys
import json
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set Hugging Face cache directory
DEFAULT_CACHE_DIR = "/cluster/scratch/yongyu/cache"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_heads import (
    ReasoningHeadDiscovery,
    ReasoningHeadEvaluator,
    generate_evaluation_report
)


def main():
    parser = argparse.ArgumentParser(
        description="Discover and evaluate reasoning heads for backward-chaining reasoning"
    )
    
    # Discovery arguments
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover reasoning heads from backward-chaining-circuits"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--backward_chaining_dir",
        type=str,
        default="../../backward-chaining-circuits",
        help="Path to backward-chaining-circuits directory"
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="ablation",
        choices=["ablation", "causal_patching", "mutual_info"],
        help="Head scoring method"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=20,
        help="Number of examples per subtask"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K heads to select per subtask"
    )
    parser.add_argument(
        "--output_heads",
        type=str,
        default="discovered_heads.json",
        help="Output file for discovered heads"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate discovered heads on benchmarks"
    )
    parser.add_argument(
        "--heads_file",
        type=str,
        default="discovered_heads.json",
        help="Input file with discovered heads"
    )
    parser.add_argument(
        "--benchmark_configs",
        type=str,
        help="YAML file with benchmark configurations"
    )
    parser.add_argument(
        "--subtask_filter",
        type=str,
        help="Filter heads by subtask name"
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        help="Number of top heads to mask"
    )
    parser.add_argument(
        "--output_results",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="evaluation_report.md",
        help="Output file for evaluation report"
    )
    
    # General arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML config file (overrides command line args)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for models (default: {DEFAULT_CACHE_DIR})"
    )
    
    args = parser.parse_args()
    
    # Set environment variables for Hugging Face cache
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Using cache directory: {args.cache_dir}")
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override args with config
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
        # Handle nested model config
        if "model" in config and "cache_dir" in config["model"]:
            args.cache_dir = config["model"]["cache_dir"]
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    print(f"Using cache directory: {args.cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For now, use a simplified model loading
    # In practice, you'd use the model loading from the main codebase
    try:
        from src.factories import get_model
        from src.configs import ModelConfigs, DecoderConfigs
        from dataclasses import dataclass
        
        @dataclass
        class ModelConfig:
            model_name_or_path: str = args.model_name
            max_seq_len: int = 4096
            max_new_tokens: int = 100
        
        model_configs = ModelConfigs(
            name="LLaMA3-8b-Instruct",
            model_type="instruct",
            configs=ModelConfig()
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
    discovery = ReasoningHeadDiscovery(
        model=model,
        tokenizer=tokenizer,
        backward_chaining_dir=args.backward_chaining_dir,
        device=args.device,
        scoring_method=args.scoring_method,
        cache_dir=args.cache_dir
    )
    
    # Discover heads
    if args.discover:
        print("\n" + "="*60)
        print("DISCOVERING REASONING HEADS")
        print("="*60)
        
        discovered_heads = discovery.discover_heads(
            n_examples_per_subtask=args.n_examples,
            top_k=args.top_k
        )
        
        discovery.save_discovered_heads(discovered_heads, args.output_heads)
        print(f"\nDiscovered {len(discovered_heads)} reasoning heads")
        print(f"Saved to {args.output_heads}")
    
    # Evaluate on benchmarks
    if args.evaluate:
        print("\n" + "="*60)
        print("EVALUATING ON BENCHMARKS")
        print("="*60)
        
        # Load discovered heads
        if os.path.exists(args.heads_file):
            discovered_heads = discovery.load_discovered_heads(args.heads_file)
            print(f"Loaded {len(discovered_heads)} discovered heads from {args.heads_file}")
        else:
            print(f"Warning: Heads file {args.heads_file} not found. Running discovery first...")
            discovered_heads = discovery.discover_heads(
                n_examples_per_subtask=args.n_examples,
                top_k=args.top_k
            )
            discovery.save_discovered_heads(discovered_heads, args.output_heads)
        
        # Load benchmark configs
        if args.benchmark_configs and os.path.exists(args.benchmark_configs):
            with open(args.benchmark_configs, 'r') as f:
                benchmark_configs = yaml.safe_load(f)
        else:
            # Default benchmark configs
            testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            benchmark_configs = {
                "hotpotqa": {
                    "main": os.path.join(testing_dir, "configs", "hotpotqa_model_config.yaml"),
                    "baseline": os.path.join(testing_dir, "configs", "hotpotqa_baseline_config.yaml")
                },
                "meqa": {
                    "main": os.path.join(testing_dir, "configs", "meqa_model_config.yaml"),
                    "baseline": os.path.join(testing_dir, "configs", "meqa_baseline_config.yaml")
                },
                "musique": {
                    "main": os.path.join(testing_dir, "configs", "musique_model_config.yaml"),
                    "baseline": os.path.join(testing_dir, "configs", "musique_baseline_config.yaml")
                }
            }
        
        # Initialize evaluator
        evaluator = ReasoningHeadEvaluator(
            discovery=discovery,
            testing_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Evaluate all benchmarks
        results = evaluator.evaluate_all_benchmarks(
            discovered_heads=discovered_heads,
            benchmark_configs=benchmark_configs,
            subtask_filter=args.subtask_filter,
            top_k_heads=args.top_k_heads
        )
        
        # Save results
        evaluator.save_results(results, args.output_results)
        
        # Generate report
        print("\n" + "="*60)
        print("GENERATING REPORT")
        print("="*60)
        
        # Convert results to format expected by reporting
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
            output_file=args.output_report,
            format="markdown"
        )
        
        print(f"\nReport saved to {args.output_report}")
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()

