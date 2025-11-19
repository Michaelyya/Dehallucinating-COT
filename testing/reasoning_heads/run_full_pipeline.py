"""
Full pipeline: Discover reasoning heads and test on benchmarks.

This script:
1. Discovers reasoning heads from CognitiveMirrors dataset
2. Saves them in DeCoReEntropy format
3. Runs baseline and DeCoReEntropy tests on HotpotQA, MEQA, and MuSiQue
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning_heads import ReasoningHeadDiscovery
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def discover_reasoning_heads(
    model_name: str,
    scoring_method: str = "ablation",
    n_examples: int = 20,
    top_k: int = 10,
    cache_dir: str = "/cluster/scratch/yongyu/cache"
):
    """Step 1: Discover reasoning heads from CognitiveMirrors."""
    print("="*80)
    print("STEP 1: DISCOVERING REASONING HEADS")
    print("="*80)
    
    # Set cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        ).eval()
    except Exception as e:
        print(f"Warning: Could not load model using factory, using direct loading: {e}")
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
        backward_chaining_dir="../CognitiveMirrors",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scoring_method=scoring_method,
        cache_dir=cache_dir
    )
    
    # Discover heads
    discovered_heads = discovery.discover_heads(
        n_examples_per_subtask=n_examples,
        top_k=top_k,
        single_subtask="logical_reasoning",
        min_score=0.0,
        min_confidence=0.0
    )
    
    # Save in standard format
    discovery.save_discovered_heads(discovered_heads, "discovered_heads.json")
    print(f"\nDiscovered {len(discovered_heads)} reasoning heads")
    
    # Save in DeCoReEntropy format
    model_base_name = model_name.split("/")[-1] if "/" in model_name else model_name
    retrieval_heads_dir = "../retrieval_heads/"
    decore_file = discovery.save_heads_for_decore(
        discovered_heads,
        output_dir=retrieval_heads_dir,
        model_name=model_base_name
    )
    
    print(f"\n✓ Reasoning heads saved to:")
    print(f"  - discovered_heads.json (full format)")
    print(f"  - {decore_file} (DeCoReEntropy format)")
    
    return discovered_heads, decore_file, retrieval_heads_dir


def run_benchmark_test(
    benchmark_name: str,
    config_path: str,
    retrieval_heads_dir: str,
    num_retrieval_heads: int = 10,
    is_baseline: bool = False
):
    """Run benchmark test with discovered reasoning heads."""
    print(f"\n{'='*80}")
    if is_baseline:
        print(f"TESTING {benchmark_name.upper()} - BASELINE")
    else:
        print(f"TESTING {benchmark_name.upper()} - DECOREENTROPY WITH REASONING HEADS")
    print(f"{'='*80}")
    
    # Load config
    config_path_abs = os.path.abspath(config_path)
    if not os.path.exists(config_path_abs):
        # Try relative to testing directory
        testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path_abs = os.path.join(testing_dir, config_path)
    
    if not os.path.exists(config_path_abs):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path_abs, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config to use discovered reasoning heads (only for DeCoRe, not baseline)
    if not is_baseline and "decoder" in config and "configs" in config["decoder"]:
        # Convert relative path to absolute
        retrieval_heads_dir_abs = os.path.abspath(retrieval_heads_dir)
        config["decoder"]["configs"]["retrieval_heads_dir"] = retrieval_heads_dir_abs
        config["decoder"]["configs"]["num_retrieval_heads"] = num_retrieval_heads
        print(f"Updated config to use:")
        print(f"  - retrieval_heads_dir: {retrieval_heads_dir_abs}")
        print(f"  - num_retrieval_heads: {num_retrieval_heads}")
    
    # Add testing directory to path for imports
    testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if testing_dir not in sys.path:
        sys.path.insert(0, testing_dir)
    
    # Import and run the appropriate tester
    if benchmark_name.lower() == "hotpotqa":
        from test_hotpotqa import HotpotQATester
        tester = HotpotQATester(config)
    elif benchmark_name.lower() == "meqa":
        from test_meqa import MEQATester
        tester = MEQATester(config)
    elif benchmark_name.lower() == "musique":
        from test_musique import MuSiQueTester
        tester = MuSiQueTester(config)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Run test
    metrics = tester.test()
    
    return metrics


def run_full_pipeline(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    scoring_method: str = "ablation",
    n_examples: int = 20,
    top_k: int = 10,
    benchmarks: list = None,
    cache_dir: str = "/cluster/scratch/yongyu/cache",
    skip_discovery: bool = False,
    discovered_heads_file: str = "discovered_heads.json"
):
    """
    Run the full pipeline: discover heads and test on benchmarks.
    
    Args:
        model_name: Model to use
        scoring_method: Head scoring method
        n_examples: Number of examples per subtask
        top_k: Top K heads to select
        benchmarks: List of benchmarks to test (default: all)
        cache_dir: Cache directory for models
        skip_discovery: Skip discovery step (use existing heads)
        discovered_heads_file: Path to existing discovered heads file
    """
    if benchmarks is None:
        benchmarks = ["hotpotqa", "meqa", "musique"]
    
    # Step 1: Discover reasoning heads (unless skipped)
    if not skip_discovery:
        discovered_heads, decore_file, retrieval_heads_dir = discover_reasoning_heads(
            model_name=model_name,
            scoring_method=scoring_method,
            n_examples=n_examples,
            top_k=top_k,
            cache_dir=cache_dir
        )
        num_retrieval_heads = len(discovered_heads)
    else:
        # Load existing heads
        print(f"Loading existing discovered heads from {discovered_heads_file}")
        with open(discovered_heads_file, 'r') as f:
            data = json.load(f)
        discovered_heads = data.get("heads", [])
        num_retrieval_heads = len(discovered_heads)
        retrieval_heads_dir = "../retrieval_heads/"
        model_base_name = model_name.split("/")[-1] if "/" in model_name else model_name
        decore_file = os.path.join(retrieval_heads_dir, f"{model_base_name}.json")
        print(f"Using {num_retrieval_heads} discovered heads from {decore_file}")
    
    # Step 2: Run baseline and DeCoReEntropy tests
    print(f"\n{'='*80}")
    print("STEP 2: RUNNING BENCHMARK TESTS")
    print(f"{'='*80}")
    
    # Get testing directory
    testing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    baseline_configs = {
        "hotpotqa": os.path.join(testing_dir, "configs", "hotpotqa_baseline_config.yaml"),
        "meqa": os.path.join(testing_dir, "configs", "meqa_baseline_config.yaml"),
        "musique": os.path.join(testing_dir, "configs", "musique_baseline_config.yaml")
    }
    
    model_configs = {
        "hotpotqa": os.path.join(testing_dir, "configs", "hotpotqa_model_config.yaml"),
        "meqa": os.path.join(testing_dir, "configs", "meqa_model_config.yaml"),
        "musique": os.path.join(testing_dir, "configs", "musique_model_config.yaml")
    }
    
    baseline_results = {}
    decore_results = {}
    
    for benchmark in benchmarks:
        benchmark_lower = benchmark.lower()
        
        # Run baseline
        if benchmark_lower in baseline_configs:
            config_path = baseline_configs[benchmark_lower]
            if os.path.exists(config_path):
                print(f"\nRunning baseline test for {benchmark}...")
                try:
                    baseline_metrics = run_benchmark_test(
                        benchmark_name=benchmark,
                        config_path=config_path,
                        retrieval_heads_dir=retrieval_heads_dir,
                        num_retrieval_heads=0,
                        is_baseline=True
                    )
                    baseline_results[benchmark] = baseline_metrics
                except Exception as e:
                    print(f"Error running baseline for {benchmark}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Baseline config not found: {config_path}")
        else:
            print(f"Warning: No baseline config for {benchmark}")
        
        # Run DeCoReEntropy with reasoning heads
        if benchmark_lower in model_configs:
            config_path = model_configs[benchmark_lower]
            if os.path.exists(config_path):
                print(f"\nRunning DeCoReEntropy test for {benchmark} with {num_retrieval_heads} reasoning heads...")
                try:
                    decore_metrics = run_benchmark_test(
                        benchmark_name=benchmark,
                        config_path=config_path,
                        retrieval_heads_dir=retrieval_heads_dir,
                        num_retrieval_heads=num_retrieval_heads,
                        is_baseline=False
                    )
                    decore_results[benchmark] = decore_metrics
                except Exception as e:
                    print(f"Error running DeCoReEntropy for {benchmark}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Model config not found: {config_path}")
        else:
            print(f"Warning: No model config for {benchmark}")
    
    # Step 4: Generate comparison report
    print(f"\n{'='*80}")
    print("STEP 4: GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    comparison_report = generate_comparison_report(
        baseline_results,
        decore_results,
        discovered_heads,
        num_retrieval_heads
    )
    
    # Save report
    report_file = "pipeline_comparison_report.md"
    with open(report_file, 'w') as f:
        f.write(comparison_report)
    
    print(f"\n✓ Full pipeline complete!")
    print(f"✓ Comparison report saved to {report_file}")
    
    return {
        "baseline_results": baseline_results,
        "decore_results": decore_results,
        "discovered_heads": len(discovered_heads),
        "report_file": report_file
    }


def generate_comparison_report(
    baseline_results: dict,
    decore_results: dict,
    discovered_heads: list,
    num_retrieval_heads: int
):
    """Generate a markdown comparison report."""
    report = []
    report.append("# Reasoning Heads Pipeline Comparison Report\n")
    report.append(f"**Number of Reasoning Heads Used**: {num_retrieval_heads}\n")
    report.append(f"**Total Heads Discovered**: {len(discovered_heads)}\n\n")
    
    report.append("## Summary\n\n")
    report.append("| Benchmark | Baseline | DeCoReEntropy | Difference |\n")
    report.append("|-----------|----------|---------------|------------|\n")
    
    for benchmark in baseline_results.keys():
        if benchmark not in decore_results:
            continue
        
        baseline = baseline_results[benchmark]
        decore = decore_results[benchmark]
        
        # Get F1 score (common metric)
        baseline_f1 = baseline.get("f1_score", baseline.get("exact_match", 0))
        decore_f1 = decore.get("f1_score", decore.get("exact_match", 0))
        diff = decore_f1 - baseline_f1
        
        report.append(f"| {benchmark.upper()} | {baseline_f1:.4f} | {decore_f1:.4f} | {diff:+.4f} |\n")
    
    report.append("\n## Detailed Results\n\n")
    
    for benchmark in baseline_results.keys():
        if benchmark not in decore_results:
            continue
        
        report.append(f"### {benchmark.upper()}\n\n")
        report.append("**Baseline Metrics:**\n")
        for key, value in baseline_results[benchmark].items():
            report.append(f"- {key}: {value:.4f}\n")
        
        report.append("\n**DeCoReEntropy Metrics:**\n")
        for key, value in decore_results[benchmark].items():
            report.append(f"- {key}: {value:.4f}\n")
        
        report.append("\n")
    
    report.append("## Discovered Reasoning Heads\n\n")
    report.append(f"Top {min(10, len(discovered_heads))} heads:\n\n")
    report.append("| Layer | Head | Score | Subtask |\n")
    report.append("|-------|------|-------|----------|\n")
    
    for head in discovered_heads[:10]:
        report.append(f"| {head.get('layer', head.layer)} | {head.get('head', head.head)} | "
                      f"{head.get('score', head.score):.4f} | {head.get('subtask', head.subtask)} |\n")
    
    return "".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Discover reasoning heads and test on benchmarks"
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
        choices=["ablation", "causal_patching", "mutual_info"],
        help="Head scoring method"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=20,
        help="Number of examples per subtask for discovery"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K heads to select"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["hotpotqa", "meqa", "musique"],
        help="Benchmarks to test"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/cluster/scratch/yongyu/cache",
        help="Cache directory for models"
    )
    parser.add_argument(
        "--skip_discovery",
        action="store_true",
        help="Skip discovery step (use existing heads)"
    )
    parser.add_argument(
        "--discovered_heads_file",
        type=str,
        default="discovered_heads.json",
        help="Path to existing discovered heads file (if skipping discovery)"
    )
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        model_name=args.model_name,
        scoring_method=args.scoring_method,
        n_examples=args.n_examples,
        top_k=args.top_k,
        benchmarks=args.benchmarks,
        cache_dir=args.cache_dir,
        skip_discovery=args.skip_discovery,
        discovered_heads_file=args.discovered_heads_file
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved to: {results['report_file']}")


if __name__ == "__main__":
    main()

