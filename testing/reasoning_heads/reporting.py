"""
Generate evaluation reports with metrics, head lists, effect sizes, and statistical significance.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


def generate_evaluation_report(
    results: Dict[str, Any],
    discovered_heads: List[Any],
    output_file: Optional[str] = None,
    format: str = "markdown"
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Dictionary of evaluation results from ReasoningHeadEvaluator
        discovered_heads: List of discovered reasoning heads
        output_file: Optional file to save report
        format: Report format ("markdown" or "html")
    
    Returns:
        Report string
    """
    if format == "markdown":
        report = _generate_markdown_report(results, discovered_heads)
    elif format == "html":
        report = _generate_html_report(results, discovered_heads)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report


def _generate_markdown_report(
    results: Dict[str, Any],
    discovered_heads: List[Any]
) -> str:
    """Generate markdown report."""
    report = []
    
    # Header
    report.append("# Reasoning Head Evaluation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # Summary
    report.append("## Executive Summary\n")
    report.append(f"- **Total Discovered Heads**: {len(discovered_heads)}")
    report.append(f"- **Benchmarks Evaluated**: {len(results)}")
    
    # Per-benchmark results
    for benchmark_name, benchmark_data in results.items():
        report.append(f"\n## {benchmark_name.upper()} Benchmark\n")
        
        baseline = benchmark_data.get("baseline", {})
        masked = benchmark_data.get("masked", {})
        comparison = benchmark_data.get("comparison", {})
        
        # Head information
        masked_heads = masked.get("masked_heads", [])
        report.append(f"### Masked Heads\n")
        report.append(f"- **Number of masked heads**: {len(masked_heads)}")
        report.append(f"- **Head list**: {masked_heads[:10]}{'...' if len(masked_heads) > 10 else ''}\n")
        
        # Metrics comparison
        report.append("### Performance Metrics\n")
        report.append("| Metric | Baseline | Masked | Change | Relative Change | Effect Size | Significant |")
        report.append("|--------|----------|--------|--------|-----------------|-------------|-------------|")
        
        metrics_comp = comparison.get("metrics_comparison", {})
        effect_sizes = comparison.get("effect_sizes", {})
        stat_tests = comparison.get("statistical_tests", {})
        
        for metric_name in metrics_comp.keys():
            metric_data = metrics_comp[metric_name]
            effect = effect_sizes.get(metric_name, {})
            stat = stat_tests.get(metric_name, {})
            
            baseline_val = metric_data.get("baseline", 0)
            masked_val = metric_data.get("masked", 0)
            abs_change = metric_data.get("absolute_change", 0)
            rel_change = metric_data.get("relative_change", 0)
            cohens_d = effect.get("cohens_d", 0)
            significant = stat.get("significant", False)
            
            report.append(
                f"| {metric_name} | {baseline_val:.4f} | {masked_val:.4f} | "
                f"{abs_change:+.4f} | {rel_change:+.2f}% | "
                f"{cohens_d:.3f} ({effect.get('interpretation', 'N/A')}) | "
                f"{'Yes' if significant else 'No'} |"
            )
        
        # Key findings
        report.append("\n### Key Findings\n")
        
        # Find metrics with largest changes
        changes = [
            (name, abs(data.get("absolute_change", 0)))
            for name, data in metrics_comp.items()
        ]
        changes.sort(key=lambda x: x[1], reverse=True)
        
        if changes:
            top_change = changes[0]
            report.append(f"- **Largest change**: {top_change[0]} "
                         f"({metrics_comp[top_change[0]]['relative_change']:+.2f}%)")
        
        # Count significant changes
        n_significant = sum(1 for stat in stat_tests.values() if stat.get("significant", False))
        report.append(f"- **Significant changes**: {n_significant} out of {len(stat_tests)} metrics")
    
    # Discovered heads summary
    report.append("\n## Discovered Reasoning Heads\n")
    report.append(f"Total: {len(discovered_heads)} heads across {len(set(h.subtask for h in discovered_heads))} subtasks\n")
    
    # Group by subtask
    by_subtask = {}
    for head in discovered_heads:
        if head.subtask not in by_subtask:
            by_subtask[head.subtask] = []
        by_subtask[head.subtask].append(head)
    
    for subtask_name, heads in by_subtask.items():
        report.append(f"### {subtask_name}\n")
        report.append(f"**Number of heads**: {len(heads)}\n")
        report.append("| Layer | Head | Score | Confidence | Method |")
        report.append("|-------|------|-------|------------|--------|")
        
        # Sort by score
        heads_sorted = sorted(heads, key=lambda x: x.score, reverse=True)
        for head in heads_sorted[:10]:  # Top 10
            report.append(
                f"| {head.layer} | {head.head} | {head.score:.4f} | "
                f"{head.confidence:.4f} | {head.method} |"
            )
        if len(heads_sorted) > 10:
            report.append(f"... and {len(heads_sorted) - 10} more")
        report.append("")
    
    # Methodology
    report.append("\n## Methodology\n")
    report.append("""
### Head Discovery
- Subtasks were automatically discovered from backward-chaining-circuits dataset and code
- Multiple scoring methods were used: ablation, causal patching, mutual information
- Heads were ranked by score and filtered by confidence thresholds

### Evaluation
- Baseline performance was measured on each benchmark
- Reasoning heads were masked using DeCoRe's block_list mechanism
- Performance was re-evaluated with masked heads
- Effect sizes (Cohen's d) and statistical significance were calculated
    """)
    
    return "\n".join(report)


def _generate_html_report(
    results: Dict[str, Any],
    discovered_heads: List[Any]
) -> str:
    """Generate HTML report."""
    # Convert markdown to HTML (simplified)
    markdown = _generate_markdown_report(results, discovered_heads)
    
    # Simple markdown to HTML conversion
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Reasoning Head Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; }}
    </style>
</head>
<body>
{_markdown_to_html(markdown)}
</body>
</html>
    """
    return html


def _markdown_to_html(markdown: str) -> str:
    """Convert markdown to HTML (simplified)."""
    html = markdown
    
    # Headers
    html = html.replace("# ", "<h1>").replace("\n# ", "</h1>\n<h1>")
    html = html.replace("## ", "<h2>").replace("\n## ", "</h2>\n<h2>")
    html = html.replace("### ", "<h3>").replace("\n### ", "</h3>\n<h3>")
    
    # Tables (keep as-is, browsers render them)
    # Bold
    html = html.replace("**", "<strong>").replace("**", "</strong>")
    
    # Lists
    html = html.replace("\n- ", "\n<li>")
    html = html.replace("\n<li>", "<ul>\n<li>")
    
    return html


def generate_summary_statistics(
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate summary statistics across all benchmarks."""
    stats = {
        "n_benchmarks": len(results),
        "benchmark_names": list(results.keys()),
        "overall_metrics": {},
        "aggregate_effect_sizes": {}
    }
    
    # Aggregate metrics across benchmarks
    all_metrics = set()
    for benchmark_data in results.values():
        comparison = benchmark_data.get("comparison", {})
        metrics_comp = comparison.get("metrics_comparison", {})
        all_metrics.update(metrics_comp.keys())
    
    for metric_name in all_metrics:
        changes = []
        effect_sizes = []
        
        for benchmark_data in results.values():
            comparison = benchmark_data.get("comparison", {})
            metrics_comp = comparison.get("metrics_comparison", {})
            effect_sizes_dict = comparison.get("effect_sizes", {})
            
            if metric_name in metrics_comp:
                changes.append(metrics_comp[metric_name].get("relative_change", 0))
            if metric_name in effect_sizes_dict:
                effect_sizes.append(effect_sizes_dict[metric_name].get("cohens_d", 0))
        
        if changes:
            stats["overall_metrics"][metric_name] = {
                "mean_change": np.mean(changes),
                "std_change": np.std(changes),
                "median_change": np.median(changes),
                "min_change": np.min(changes),
                "max_change": np.max(changes)
            }
        
        if effect_sizes:
            stats["aggregate_effect_sizes"][metric_name] = {
                "mean_cohens_d": np.mean(effect_sizes),
                "std_cohens_d": np.std(effect_sizes)
            }
    
    return stats

