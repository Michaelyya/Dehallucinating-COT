"""
Batch Experiment Runner for MEQA Testing
Runs multiple experiments with different configurations
"""

import os
import json
import subprocess
import argparse
from typing import List, Dict, Any
import yaml
from datetime import datetime


class ExperimentRunner:
    """
    Runs batch experiments on MEQA dataset
    """
    
    def __init__(self, base_config_dir: str = "configs"):
        self.base_config_dir = base_config_dir
        self.results_dir = "outputs"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_single_experiment(
        self, 
        config_path: str, 
        output_suffix: str = "",
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Run a single experiment
        
        Args:
            config_path: Path to configuration file
            output_suffix: Suffix to add to output directory
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with experiment results
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = os.path.basename(config_path).replace('.yaml', '')
        output_dir = os.path.join(
            self.results_dir, 
            f"{config_name}_{timestamp}{output_suffix}"
        )
        
        # Run the experiment
        cmd = [
            "python", "test_meqa.py",
            "--config", config_path,
            "--output_dir", output_dir,
            "--num_samples", str(num_samples)
        ]
        
        print(f"Running experiment: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Experiment completed successfully: {output_dir}")
            
            # Load results
            metrics_file = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return {
                    "status": "success",
                    "output_dir": output_dir,
                    "metrics": metrics,
                    "config": config_path
                }
            else:
                return {
                    "status": "success_no_metrics",
                    "output_dir": output_dir,
                    "config": config_path
                }
        
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed: {e}")
            print(f"Error output: {e.stderr}")
            return {
                "status": "failed",
                "error": str(e),
                "config": config_path
            }
    
    def run_baseline_experiments(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Run baseline experiments
        """
        results = []
        
        # Baseline model
        baseline_config = os.path.join(self.base_config_dir, "baseline_config.yaml")
        if os.path.exists(baseline_config):
            result = self.run_single_experiment(
                baseline_config, 
                "_baseline", 
                num_samples
            )
            results.append(result)
        
        return results
    
    def run_decore_experiments(
        self, 
        num_retrieval_heads: List[int] = [5, 10, 20, 50],
        num_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Run DeCoRe experiments with different numbers of retrieval heads
        """
        results = []
        
        for num_heads in num_retrieval_heads:
            # Create temporary config with specific number of heads
            temp_config = self._create_temp_config(num_heads)
            
            result = self.run_single_experiment(
                temp_config,
                f"_decore_{num_heads}heads",
                num_samples
            )
            results.append(result)
            
            # Clean up temp config
            os.remove(temp_config)
        
        return results
    
    def _create_temp_config(self, num_retrieval_heads: int) -> str:
        """
        Create temporary configuration file with specific number of retrieval heads
        """
        base_config_path = os.path.join(self.base_config_dir, "meqa_config.yaml")
        
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify the number of retrieval heads
        config["decoder"]["configs"]["num_retrieval_heads"] = num_retrieval_heads
        
        # Create temporary file
        temp_config_path = f"temp_config_{num_retrieval_heads}heads.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        return temp_config_path
    
    def run_comparison_experiments(self, num_samples: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run comprehensive comparison experiments
        """
        print("Starting comprehensive MEQA experiments...")
        
        all_results = {
            "baseline": [],
            "decore": [],
            "summary": {}
        }
        
        # Run baseline experiments
        print("\n=== Running Baseline Experiments ===")
        baseline_results = self.run_baseline_experiments(num_samples)
        all_results["baseline"] = baseline_results
        
        # Run DeCoRe experiments
        print("\n=== Running DeCoRe Experiments ===")
        decore_results = self.run_decore_experiments(num_samples=num_samples)
        all_results["decore"] = decore_results
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results)
        
        # Save comprehensive results
        summary_file = os.path.join(self.results_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll experiments completed. Summary saved to {summary_file}")
        return all_results
    
    def _generate_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate summary of all experiments
        """
        summary = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "best_baseline": None,
            "best_decore": None,
            "comparison": {}
        }
        
        # Process baseline results
        baseline_metrics = []
        for result in results["baseline"]:
            summary["total_experiments"] += 1
            if result["status"] == "success":
                summary["successful_experiments"] += 1
                if "metrics" in result:
                    baseline_metrics.append(result["metrics"])
            else:
                summary["failed_experiments"] += 1
        
        # Process DeCoRe results
        decore_metrics = []
        for result in results["decore"]:
            summary["total_experiments"] += 1
            if result["status"] == "success":
                summary["successful_experiments"] += 1
                if "metrics" in result:
                    decore_metrics.append(result["metrics"])
            else:
                summary["failed_experiments"] += 1
        
        # Find best results
        if baseline_metrics:
            best_baseline = max(baseline_metrics, key=lambda x: x.get("exact_match", 0))
            summary["best_baseline"] = best_baseline
        
        if decore_metrics:
            best_decore = max(decore_metrics, key=lambda x: x.get("exact_match", 0))
            summary["best_decore"] = best_decore
        
        # Generate comparison
        if baseline_metrics and decore_metrics:
            avg_baseline = self._average_metrics(baseline_metrics)
            avg_decore = self._average_metrics(decore_metrics)
            
            summary["comparison"] = {
                "baseline_avg": avg_baseline,
                "decore_avg": avg_decore,
                "improvement": {
                    key: avg_decore.get(key, 0) - avg_baseline.get(key, 0)
                    for key in avg_baseline.keys()
                }
            }
        
        return summary
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across multiple experiments
        """
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key != "total_samples":  # Skip non-numeric fields
                values = [m.get(key, 0) for m in metrics_list]
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    def run_attention_analysis(self, results_dir: str = None):
        """
        Run attention analysis on all completed experiments
        """
        if results_dir is None:
            results_dir = self.results_dir
        
        print("Running attention analysis on completed experiments...")
        
        # Find all detailed results files
        detailed_results_files = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file == "detailed_results.json":
                    detailed_results_files.append(os.path.join(root, file))
        
        if not detailed_results_files:
            print("No detailed results files found for attention analysis")
            return
        
        # Run attention analysis for each experiment
        for results_file in detailed_results_files:
            print(f"Analyzing attention for: {results_file}")
            
            # Create attention analysis output directory
            analysis_dir = os.path.dirname(results_file)
            
            # Run attention analysis
            cmd = [
                "python", "attention_analysis.py",
                "--results_file", results_file,
                "--output_dir", analysis_dir
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"Attention analysis completed for: {analysis_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Attention analysis failed for {results_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run batch MEQA experiments")
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=["baseline", "decore", "comparison", "attention"],
        default="comparison",
        help="Type of experiments to run"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Number of retrieval heads for DeCoRe experiments"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    if args.experiment_type == "baseline":
        results = runner.run_baseline_experiments(args.num_samples)
    elif args.experiment_type == "decore":
        results = runner.run_decore_experiments(args.num_heads, args.num_samples)
    elif args.experiment_type == "comparison":
        results = runner.run_comparison_experiments(args.num_samples)
    elif args.experiment_type == "attention":
        runner.run_attention_analysis(args.results_dir)
        return
    
    # Print summary
    print("\n=== Experiment Summary ===")
    if isinstance(results, dict):
        summary = results.get("summary", {})
        print(f"Total experiments: {summary.get('total_experiments', 0)}")
        print(f"Successful: {summary.get('successful_experiments', 0)}")
        print(f"Failed: {summary.get('failed_experiments', 0)}")
        
        if summary.get("best_baseline"):
            print(f"Best baseline exact match: {summary['best_baseline'].get('exact_match', 0):.4f}")
        
        if summary.get("best_decore"):
            print(f"Best DeCoRe exact match: {summary['best_decore'].get('exact_match', 0):.4f}")
        
        if summary.get("comparison"):
            improvement = summary["comparison"]["improvement"]
            print(f"Average improvement in exact match: {improvement.get('exact_match', 0):.4f}")
    else:
        print(f"Completed {len(results)} experiments")


if __name__ == "__main__":
    main()