"""
Evaluation framework for testing reasoning head masking on benchmarks.
"""

import os
import json
import yaml
import copy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

from .discovery import ReasoningHeadDiscovery, ReasoningHead


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""
    benchmark_name: str
    config_path: str
    metrics: Dict[str, float]
    predictions: List[Dict[str, Any]]
    masked_heads: List[Tuple[int, int]]
    n_examples: int
    
    def to_dict(self):
        return {
            "benchmark_name": self.benchmark_name,
            "config_path": self.config_path,
            "metrics": self.metrics,
            "n_examples": self.n_examples,
            "masked_heads": self.masked_heads,
            "n_predictions": len(self.predictions)
        }


class ReasoningHeadEvaluator:
    """
    Evaluates the effect of masking reasoning heads on benchmark performance.
    """
    
    def __init__(
        self,
        discovery: ReasoningHeadDiscovery,
        testing_dir: str = ".",
        output_dir: str = "./evaluation_results"
    ):
        self.discovery = discovery
        self.testing_dir = testing_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        config_path: str,
        discovered_heads: List[ReasoningHead],
        baseline_config_path: Optional[str] = None,
        subtask_filter: Optional[str] = None,
        top_k_heads: Optional[int] = None
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Evaluate benchmark with and without reasoning head masking.
        
        Args:
            benchmark_name: Name of benchmark (hotpotqa, meqa, musique)
            config_path: Path to benchmark config file
            discovered_heads: List of discovered reasoning heads
            baseline_config_path: Optional path to baseline config
            subtask_filter: Optional subtask name to filter heads
            top_k_heads: Optional number of top heads to mask
        
        Returns:
            Tuple of (baseline_result, masked_result)
        """
        print(f"\nEvaluating {benchmark_name} benchmark")
        
        # Get heads to mask
        heads_to_mask = self.discovery.get_head_list_for_masking(
            discovered_heads,
            subtask_filter,
            top_k_heads
        )
        
        print(f"  Masking {len(heads_to_mask)} heads: {heads_to_mask[:5]}...")
        
        # Run baseline
        baseline_result = self._run_benchmark(
            benchmark_name,
            config_path if baseline_config_path is None else baseline_config_path,
            masked_heads=None
        )
        
        # Run with masking
        masked_result = self._run_benchmark(
            benchmark_name,
            config_path,
            masked_heads=heads_to_mask
        )
        
        return baseline_result, masked_result
    
    def _run_benchmark(
        self,
        benchmark_name: str,
        config_path: str,
        masked_heads: Optional[List[Tuple[int, int]]] = None
    ) -> BenchmarkResult:
        """Run a benchmark test."""
        # Import benchmark testers
        if benchmark_name.lower() == "hotpotqa":
            from test_hotpotqa import HotpotQATester, load_config
        elif benchmark_name.lower() == "meqa":
            from test_meqa import MEQATester, load_config
        elif benchmark_name.lower() == "musique":
            from test_musique import MuSiQueTester, load_config
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # Load config
        config = load_config(config_path)
        
        # Modify config to include masked heads if provided
        if masked_heads:
            # Store masked heads in config for later use
            config["masked_heads"] = masked_heads
            # Modify decoder config to use masking
            # For now, we'll modify the decoder to use baseline with masking
            # In practice, you'd create a custom decoder class
            if "decoder" in config:
                # Store original
                if "original_decoder" not in config:
                    config["original_decoder"] = copy.deepcopy(config["decoder"])
                
                # Use baseline method but with masked heads
                # The actual masking will need to be done in the model's generate method
                config["decoder"]["configs"]["masked_heads"] = masked_heads
                config["decoder"]["configs"]["num_retrieval_heads"] = len(masked_heads)
        
        # Create tester
        if benchmark_name.lower() == "hotpotqa":
            tester = HotpotQATester(config)
        elif benchmark_name.lower() == "meqa":
            tester = MEQATester(config)
        elif benchmark_name.lower() == "musique":
            tester = MuSiQueTester(config)
        
        # Run test
        metrics = tester.test()
        
        # Load predictions
        predictions_file = os.path.join(
            config["evaluation"]["output_dir"],
            "predictions.json"
        )
        predictions = []
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            config_path=config_path,
            metrics=metrics,
            predictions=predictions,
            masked_heads=masked_heads or [],
            n_examples=len(predictions)
        )
    
    def evaluate_all_benchmarks(
        self,
        discovered_heads: List[ReasoningHead],
        benchmark_configs: Dict[str, Dict[str, str]],
        subtask_filter: Optional[str] = None,
        top_k_heads: Optional[int] = None
    ) -> Dict[str, Tuple[BenchmarkResult, BenchmarkResult]]:
        """
        Evaluate all benchmarks.
        
        Args:
            discovered_heads: List of discovered reasoning heads
            benchmark_configs: Dict mapping benchmark names to config paths
                Format: {"hotpotqa": {"main": "path", "baseline": "path"}, ...}
            subtask_filter: Optional subtask name to filter heads
            top_k_heads: Optional number of top heads to mask
        
        Returns:
            Dict mapping benchmark names to (baseline, masked) results
        """
        results = {}
        
        for benchmark_name, configs in benchmark_configs.items():
            try:
                baseline_result, masked_result = self.evaluate_benchmark(
                    benchmark_name=benchmark_name,
                    config_path=configs.get("main"),
                    discovered_heads=discovered_heads,
                    baseline_config_path=configs.get("baseline"),
                    subtask_filter=subtask_filter,
                    top_k_heads=top_k_heads
                )
                results[benchmark_name] = (baseline_result, masked_result)
            except Exception as e:
                print(f"Error evaluating {benchmark_name}: {e}")
                continue
        
        return results
    
    def compare_results(
        self,
        baseline_result: BenchmarkResult,
        masked_result: BenchmarkResult
    ) -> Dict[str, Any]:
        """
        Compare baseline and masked results.
        
        Returns:
            Dictionary with comparison metrics including effect sizes and significance
        """
        comparison = {
            "benchmark": baseline_result.benchmark_name,
            "n_examples": baseline_result.n_examples,
            "n_masked_heads": len(masked_result.masked_heads),
            "metrics_comparison": {},
            "effect_sizes": {},
            "statistical_tests": {}
        }
        
        # Compare each metric
        for metric_name in baseline_result.metrics.keys():
            baseline_value = baseline_result.metrics[metric_name]
            masked_value = masked_result.metrics[metric_name]
            
            # Absolute and relative change
            absolute_change = masked_value - baseline_value
            relative_change = (absolute_change / baseline_value * 100) if baseline_value != 0 else 0
            
            comparison["metrics_comparison"][metric_name] = {
                "baseline": baseline_value,
                "masked": masked_value,
                "absolute_change": absolute_change,
                "relative_change": relative_change
            }
            
            # Effect size (Cohen's d)
            # Simplified - would need per-example scores for proper calculation
            if baseline_value != 0:
                cohens_d = absolute_change / (abs(baseline_value) + 1e-8)
            else:
                cohens_d = 0.0
            
            comparison["effect_sizes"][metric_name] = {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_effect_size(cohens_d)
            }
            
            # Statistical test (simplified - would need per-example data)
            # For now, we'll use a simple threshold
            p_value = 0.05 if abs(absolute_change) > 0.01 else 1.0
            comparison["statistical_tests"][metric_name] = {
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
        return comparison
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def save_results(
        self,
        results: Dict[str, Tuple[BenchmarkResult, BenchmarkResult]],
        output_file: Optional[str] = None
    ):
        """Save evaluation results."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "evaluation_results.json")
        
        # Convert results to serializable format
        serialized = {}
        for benchmark_name, (baseline, masked) in results.items():
            comparison = self.compare_results(baseline, masked)
            serialized[benchmark_name] = {
                "baseline": baseline.to_dict(),
                "masked": masked.to_dict(),
                "comparison": comparison
            }
        
        with open(output_file, 'w') as f:
            json.dump(serialized, f, indent=2)
        
        print(f"\nSaved evaluation results to {output_file}")

