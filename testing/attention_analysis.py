import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import torch
from collections import defaultdict
import pandas as pd


class AttentionAnalyzer:
    def __init__(self, results_path: str):
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.attention_data = []
        self._extract_attention_data()
    
    def _extract_attention_data(self):
        for result in self.results:
            if "alphas" in result and result["alphas"] is not None:
                # Convert alphas to numpy array if it's a list
                if isinstance(result["alphas"], list):
                    alphas = np.array(result["alphas"])
                else:
                    alphas = result["alphas"]
                
                self.attention_data.append({
                    "example_id": result["sample_id"],
                    "alphas": alphas,
                    "exact_match": result["exact_match"],
                    "f1_score": result["f1_score"],
                    "explanation_f1": result["explanation_f1"],
                    "predicted_answer": result["predicted_answer"],
                    "reference_answer": result["reference_answer"],
                })
    
    def analyze_attention_by_correctness(self) -> Dict[str, Any]:
        """
        Analyze attention patterns by answer correctness
        """
        correct_attentions = []
        incorrect_attentions = []
        
        for data in self.attention_data:
            if data["exact_match"]:
                correct_attentions.append(data["alphas"])
            else:
                incorrect_attentions.append(data["alphas"])
        
        if not correct_attentions or not incorrect_attentions:
            return {"error": "Not enough data for comparison"}
        
        # Calculate statistics
        correct_mean = np.mean(correct_attentions, axis=0)
        incorrect_mean = np.mean(incorrect_attentions, axis=0)
        
        # Calculate difference
        attention_diff = correct_mean - incorrect_mean
        
        return {
            "correct_mean": correct_mean.tolist(),
            "incorrect_mean": incorrect_mean.tolist(),
            "attention_difference": attention_diff.tolist(),
            "correct_count": len(correct_attentions),
            "incorrect_count": len(incorrect_attentions),
        }
    
    def identify_hallucination_heads(self, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Identify attention heads that are most associated with hallucination
        
        Args:
            threshold: Threshold for considering a head as significant
            
        Returns:
            Dictionary with hallucination-related head analysis
        """
        analysis = self.analyze_attention_by_correctness()
        
        if "error" in analysis:
            return analysis
        
        attention_diff = np.array(analysis["attention_difference"])
        
        # Find heads with significant differences
        significant_heads = np.where(np.abs(attention_diff) > threshold)[0]
        
        # Categorize heads
        hallucination_heads = np.where(attention_diff < -threshold)[0]  # Higher attention in incorrect
        faithfulness_heads = np.where(attention_diff > threshold)[0]    # Higher attention in correct
        
        return {
            "total_heads": len(attention_diff),
            "significant_heads": significant_heads.tolist(),
            "hallucination_heads": hallucination_heads.tolist(),
            "faithfulness_heads": faithfulness_heads.tolist(),
            "attention_differences": attention_diff.tolist(),
            "threshold": threshold,
        }
    
    def plot_attention_patterns(self, save_path: Optional[str] = None):
        """
        Plot attention patterns for correct vs incorrect answers
        """
        analysis = self.analyze_attention_by_correctness()
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean attention for correct answers
        axes[0, 0].plot(analysis["correct_mean"])
        axes[0, 0].set_title("Mean Attention - Correct Answers")
        axes[0, 0].set_xlabel("Generation Step")
        axes[0, 0].set_ylabel("Attention Weight")
        
        # Plot 2: Mean attention for incorrect answers
        axes[0, 1].plot(analysis["incorrect_mean"])
        axes[0, 1].set_title("Mean Attention - Incorrect Answers")
        axes[0, 1].set_xlabel("Generation Step")
        axes[0, 1].set_ylabel("Attention Weight")
        
        # Plot 3: Difference in attention
        axes[1, 0].plot(analysis["attention_difference"])
        axes[1, 0].set_title("Attention Difference (Correct - Incorrect)")
        axes[1, 0].set_xlabel("Generation Step")
        axes[1, 0].set_ylabel("Attention Difference")
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 4: Distribution of attention differences
        axes[1, 1].hist(analysis["attention_difference"], bins=20, alpha=0.7)
        axes[1, 1].set_title("Distribution of Attention Differences")
        axes[1, 1].set_xlabel("Attention Difference")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_attention_by_f1_score(self, bins: int = 5) -> Dict[str, Any]:
        """
        Analyze attention patterns by F1 score bins
        """
        # Bin the F1 scores
        f1_scores = [data["f1_score"] for data in self.attention_data]
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(f1_scores, bin_edges) - 1
        
        # Group attention by F1 bins
        binned_attentions = defaultdict(list)
        for i, data in enumerate(self.attention_data):
            bin_idx = bin_indices[i]
            binned_attentions[bin_idx].append(data["alphas"])
        
        # Calculate mean attention for each bin
        bin_means = {}
        bin_counts = {}
        for bin_idx, attentions in binned_attentions.items():
            if attentions:
                bin_means[bin_idx] = np.mean(attentions, axis=0).tolist()
                bin_counts[bin_idx] = len(attentions)
        
        return {
            "bin_edges": bin_edges.tolist(),
            "bin_means": bin_means,
            "bin_counts": bin_counts,
        }
    
    def plot_f1_attention_correlation(self, save_path: Optional[str] = None):
        """
        Plot correlation between F1 scores and attention patterns
        """
        f1_scores = [data["f1_score"] for data in self.attention_data]
        attentions = [data["alphas"] for data in self.attention_data]
        
        # Calculate correlation for each attention step
        correlations = []
        for step in range(len(attentions[0])):
            step_attentions = [att[step] for att in attentions]
            corr = np.corrcoef(f1_scores, step_attentions)[0, 1]
            correlations.append(corr)
        
        plt.figure(figsize=(12, 6))
        plt.plot(correlations)
        plt.title("Correlation between F1 Score and Attention Weights")
        plt.xlabel("Generation Step")
        plt.ylabel("Correlation Coefficient")
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_attention_report(self, output_path: str):
        """
        Generate a comprehensive attention analysis report
        """
        report = {
            "summary": {
                "total_samples": len(self.attention_data),
                "samples_with_attention": len(self.attention_data),
            },
            "correctness_analysis": self.analyze_attention_by_correctness(),
            "hallucination_heads": self.identify_hallucination_heads(),
            "f1_analysis": self.analyze_attention_by_f1_score(),
        }
        
        # Add statistics
        if self.attention_data:
            all_attentions = [data["alphas"] for data in self.attention_data]
            report["statistics"] = {
                "mean_attention": np.mean(all_attentions).tolist(),
                "std_attention": np.std(all_attentions).tolist(),
                "min_attention": np.min(all_attentions).tolist(),
                "max_attention": np.max(all_attentions).tolist(),
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Attention analysis report saved to {output_path}")
        return report


def compare_models_attention(results_paths: Dict[str, str], output_dir: str):
    """
    Compare attention patterns between different models
    
    Args:
        results_paths: Dictionary mapping model names to their results paths
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_analyses = {}
    for model_name, results_path in results_paths.items():
        analyzer = AttentionAnalyzer(results_path)
        model_analyses[model_name] = analyzer
    
    # Compare hallucination heads
    hallucination_comparison = {}
    for model_name, analyzer in model_analyses.items():
        hallucination_heads = analyzer.identify_hallucination_heads()
        hallucination_comparison[model_name] = {
            "hallucination_heads": hallucination_heads.get("hallucination_heads", []),
            "faithfulness_heads": hallucination_heads.get("faithfulness_heads", []),
            "total_heads": hallucination_heads.get("total_heads", 0),
        }
    
    # Save comparison
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(hallucination_comparison, f, indent=2)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Number of hallucination heads
    model_names = list(hallucination_comparison.keys())
    hallucination_counts = [len(hallucination_comparison[name]["hallucination_heads"]) for name in model_names]
    faithfulness_counts = [len(hallucination_comparison[name]["faithfulness_heads"]) for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, hallucination_counts, width, label='Hallucination Heads', alpha=0.7)
    axes[0, 0].bar(x + width/2, faithfulness_counts, width, label='Faithfulness Heads', alpha=0.7)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Number of Heads')
    axes[0, 0].set_title('Attention Head Classification by Model')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    
    # Plot 2: Attention patterns comparison
    for i, (model_name, analyzer) in enumerate(model_analyses.items()):
        analysis = analyzer.analyze_attention_by_correctness()
        if "error" not in analysis:
            axes[0, 1].plot(analysis["attention_difference"], label=model_name, alpha=0.7)
    
    axes[0, 1].set_title('Attention Difference Patterns')
    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('Attention Difference')
    axes[0, 1].legend()
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    comparison_plot = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison saved to {output_dir}")
    return hallucination_comparison


if __name__ == "__main__":
    # Example usage
    results_path = "testing/outputs/detailed_results.json"
    
    if os.path.exists(results_path):
        analyzer = AttentionAnalyzer(results_path)
        
        # Generate analysis
        report = analyzer.generate_attention_report("testing/outputs/attention_report.json")
        
        # Create plots
        analyzer.plot_attention_patterns("testing/outputs/attention_patterns.png")
        analyzer.plot_f1_attention_correlation("testing/outputs/f1_correlation.png")
        
        print("Attention analysis completed!")
    else:
        print(f"Results file not found: {results_path}")
