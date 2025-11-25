"""
Script to run benchmarks (HotpotQA, MEQA, MuSiQue) with discovered reasoning heads using DeCoRe.

Usage:
    python run_benchmarks_with_discovered_heads.py \
        --model_name "Qwen/Qwen3-4B-Instruct-2507" \
        --num_retrieval_heads 20 \
        --benchmarks hotpotqa meqa musique \
        --num_samples 100
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path

# Add testing directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_hotpotqa import HotpotQATester, load_config as load_hotpotqa_config
from test_meqa import MEQATester, load_config as load_meqa_config
from test_musique import MuSiQueTester, load_config as load_musique_config


def update_config_for_model(
    config_path: str,
    model_name: str,
    num_retrieval_heads: int,
    retrieval_heads_dir: str = "../retrieval_heads/",
    output_suffix: str = None
) -> str:
    """
    Update a benchmark config file to use the specified model and discovered heads.
    
    Args:
        config_path: Path to the original config file
        model_name: Model name (e.g., "Qwen/Qwen3-4B-Instruct-2507")
        num_retrieval_heads: Number of retrieval heads to use
        retrieval_heads_dir: Directory containing the discovered heads JSON file
        output_suffix: Optional suffix for output directory
    
    Returns:
        Path to the updated config file
    """
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model name
    model_base_name = model_name.split("/")[-1] if "/" in model_name else model_name
    config["model"]["name"] = model_base_name
    config["model"]["configs"]["model_name_or_path"] = model_name
    
    # Update decoder config for DeCoRe
    if "decoder" in config and "decore" in config["decoder"]["method"].lower():
        config["decoder"]["configs"]["retrieval_heads_dir"] = retrieval_heads_dir
        config["decoder"]["configs"]["num_retrieval_heads"] = num_retrieval_heads
    
    # Update output directory
    if output_suffix:
        original_output = config["evaluation"]["output_dir"]
        config["evaluation"]["output_dir"] = f"{original_output}_{output_suffix}"
    else:
        # Add model name to output dir
        original_output = config["evaluation"]["output_dir"]
        config["evaluation"]["output_dir"] = f"{original_output}_{model_base_name}"
    
    # Save updated config
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    updated_config_path = os.path.join(
        config_dir,
        f"{config_name.replace('.yaml', '')}_{model_base_name}.yaml"
    )
    
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated config saved to: {updated_config_path}")
    return updated_config_path


def verify_heads_file(model_name: str, retrieval_heads_dir: str = "../retrieval_heads/") -> bool:
    """Verify that the discovered heads file exists."""
    model_base_name = model_name.split("/")[-1] if "/" in model_name else model_name
    heads_file = os.path.join(retrieval_heads_dir, f"{model_base_name}.json")
    
    if not os.path.exists(heads_file):
        print(f"ERROR: Heads file not found: {heads_file}")
        print(f"  Expected file: {heads_file}")
        print(f"  Please run discovery first:")
        print(f"    python discover_reasoning_heads.py --model_name {model_name}")
        return False
    
    # Verify file format
    try:
        with open(heads_file, 'r') as f:
            heads_data = json.loads(f.readline())
        
        if not isinstance(heads_data, dict):
            print(f"ERROR: Invalid heads file format. Expected dict, got {type(heads_data)}")
            return False
        
        num_heads_in_file = len(heads_data)
        print(f"✓ Found {num_heads_in_file} heads in {heads_file}")
        print(f"  Sample heads: {list(heads_data.keys())[:5]}")
        return True
        
    except Exception as e:
        print(f"ERROR: Could not read heads file: {e}")
        return False


def run_benchmark(benchmark_name: str, config_path: str) -> dict:
    """Run a single benchmark and return metrics."""
    print(f"\n{'='*80}")
    print(f"Running {benchmark_name.upper()} benchmark")
    print(f"{'='*80}")
    
    if benchmark_name.lower() == "hotpotqa":
        config = load_hotpotqa_config(config_path)
        tester = HotpotQATester(config)
    elif benchmark_name.lower() == "meqa":
        config = load_meqa_config(config_path)
        tester = MEQATester(config)
    elif benchmark_name.lower() == "musique":
        config = load_musique_config(config_path)
        tester = MuSiQueTester(config)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    metrics = tester.test()
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks with discovered reasoning heads using DeCoRe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with discovered heads
  python run_benchmarks_with_discovered_heads.py \\
      --model_name "Qwen/Qwen3-4B-Instruct-2507" \\
      --num_retrieval_heads 20 \\
      --benchmarks hotpotqa meqa musique

  # Run only HotpotQA with 100 samples
  python run_benchmarks_with_discovered_heads.py \\
      --model_name "Qwen/Qwen3-4B-Instruct-2507" \\
      --num_retrieval_heads 20 \\
      --benchmarks hotpotqa \\
      --num_samples 100
        """
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-4B-Instruct-2507')"
    )
    parser.add_argument(
        "--num_retrieval_heads",
        type=int,
        required=True,
        help="Number of retrieval heads to use (should match discovered heads)"
    )
    parser.add_argument(
        "--retrieval_heads_dir",
        type=str,
        default="../retrieval_heads/",
        help="Directory containing discovered heads JSON file"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=["hotpotqa", "meqa", "musique"],
        default=["hotpotqa", "meqa", "musique"],
        help="Benchmarks to run"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Override number of samples (default: use config value)"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing benchmark config files"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run baseline (no DeCoRe) for comparison"
    )
    
    args = parser.parse_args()
    
    # Verify heads file exists
    print(f"\n{'='*80}")
    print("VERIFYING DISCOVERED HEADS")
    print(f"{'='*80}")
    if not verify_heads_file(args.model_name, args.retrieval_heads_dir):
        return 1
    
    # Get absolute path to config directory
    testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(testing_dir, args.config_dir)
    
    # Map benchmark names to config files
    benchmark_configs = {
        "hotpotqa": os.path.join(config_dir, "hotpotqa_model_config.yaml"),
        "meqa": os.path.join(config_dir, "meqa_model_config.yaml"),
        "musique": os.path.join(config_dir, "musique_model_config.yaml"),
    }
    
    all_results = {}
    
    # Run each benchmark
    for benchmark in args.benchmarks:
        if benchmark not in benchmark_configs:
            print(f"Warning: No config found for {benchmark}, skipping")
            continue
        
        config_path = benchmark_configs[benchmark]
        
        # Update config for this model
        updated_config = update_config_for_model(
            config_path,
            args.model_name,
            args.num_retrieval_heads,
            args.retrieval_heads_dir
        )
        
        # Override num_samples if specified
        if args.num_samples:
            with open(updated_config, 'r') as f:
                config = yaml.safe_load(f)
            config["data"]["num_samples"] = args.num_samples
            with open(updated_config, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Run benchmark
        try:
            metrics = run_benchmark(benchmark, updated_config)
            all_results[benchmark] = metrics
        except Exception as e:
            print(f"ERROR: Failed to run {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Retrieval Heads: {args.num_retrieval_heads}")
    print(f"\nResults:")
    for benchmark, metrics in all_results.items():
        print(f"\n{benchmark.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

