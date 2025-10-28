# MEQA Testing Framework

This framework provides tools for testing DeCoRe and baseline models on the MEQA (Multi-hop Event-centric Question Answering) dataset, with a focus on analyzing attention patterns and identifying hallucination-related heads.

## Overview

The framework includes:
- **MEQA Dataset Integration**: Custom dataset class for loading and processing MEQA data
- **Evaluation Metrics**: Comprehensive metrics for answer accuracy and reasoning quality
- **Attention Analysis**: Tools for analyzing attention patterns and identifying hallucination-related heads
- **Testing Scripts**: Easy-to-use scripts for running experiments

## Directory Structure

```
testing/
├── README.md                    # This file
├── meqa_dataset.py             # MEQA dataset class
├── meqa_metrics.py             # Evaluation metrics
├── test_meqa.py                # Main testing script
├── attention_analysis.py       # Attention analysis tools
├── run_experiments.py          # Batch experiment runner
├── configs/                    # Configuration files
│   ├── meqa_config.yaml       # DeCoRe configuration
│   └── baseline_config.yaml   # Baseline configuration
└── outputs/                    # Experiment outputs (created automatically)
```

## Quick Start

### 1. Test DeCoRe on MEQA

```bash
cd /Users/yonganyu/Desktop/decore/testing
python test_meqa.py --config configs/meqa_config.yaml --num_samples 50
```

### 2. Test Baseline Model

```bash
python test_meqa.py --config configs/baseline_config.yaml --num_samples 50
```

### 3. Run Batch Experiments

```bash
python run_experiments.py
```

### 4. Analyze Attention Patterns

```bash
python attention_analysis.py
```

## Configuration

### MEQA Configuration (`configs/meqa_config.yaml`)

Key parameters:
- `data.num_samples`: Number of samples to test (-1 for all)
- `data.split`: Dataset split (train/dev/test)
- `decoder.configs.num_retrieval_heads`: Number of retrieval heads for DeCoRe
- `model.configs.max_new_tokens`: Maximum tokens for generation
- `evaluation.save_predictions`: Whether to save detailed predictions

### Baseline Configuration (`configs/baseline_config.yaml`)

Similar to MEQA config but with baseline decoder settings.

## Evaluation Metrics

The framework provides several evaluation metrics:

1. **Answer Accuracy**:
   - `exact_match`: Exact string match between predicted and reference answers
   - `f1_score`: F1 score on answer tokens

2. **Reasoning Quality**:
   - `explanation_f1`: F1 score on explanation tokens
   - `explanation_bleu`: BLEU-like score for explanation quality

## Attention Analysis

The attention analysis tools help identify:

1. **Hallucination Heads**: Attention heads that are more active during incorrect reasoning
2. **Faithfulness Heads**: Attention heads that are more active during correct reasoning
3. **Attention Patterns**: How attention changes during generation for correct vs incorrect answers

### Key Analysis Functions

- `identify_hallucination_heads()`: Find heads associated with hallucination
- `analyze_attention_by_correctness()`: Compare attention patterns for correct/incorrect answers
- `plot_attention_patterns()`: Visualize attention differences
- `generate_attention_report()`: Create comprehensive analysis report

## Output Files

Each experiment generates:

- `predictions.json`: Raw model predictions
- `metrics.json`: Overall evaluation metrics
- `detailed_results.json`: Per-sample detailed results
- `attention_report.json`: Attention analysis report (if attention data available)
- `attention_patterns.png`: Attention visualization plots

## Research Applications

This framework supports research on:

1. **Faithfulness in Reasoning**: Understanding which attention heads contribute to faithful vs unfaithful reasoning
2. **Attention Steering**: Post-hoc control of attention heads to improve reasoning quality
3. **Multi-hop Reasoning**: Analysis of how models handle complex event-centric questions
4. **CoT Analysis**: Understanding chain-of-thought reasoning patterns

## Example Research Questions

1. **Can we identify attention heads responsible for hallucination?**
   - Use `identify_hallucination_heads()` to find problematic heads

2. **How does attention differ between correct and incorrect reasoning?**
   - Use `analyze_attention_by_correctness()` to compare patterns

3. **Can we steer attention to improve reasoning quality?**
   - Modify attention weights of identified heads during inference

4. **Which models are more prone to hallucination?**
   - Compare attention patterns across different models

## Customization

### Adding New Metrics

Extend `MEQAMetrics` class in `meqa_metrics.py`:

```python
def custom_metric(self, pred, ref):
    # Your custom metric implementation
    return score

def compute_metrics(self):
    metrics = super().compute_metrics()
    metrics["custom_metric"] = self.custom_metric(pred, ref)
    return metrics
```

### Adding New Attention Analysis

Extend `AttentionAnalyzer` class in `attention_analysis.py`:

```python
def custom_attention_analysis(self):
    # Your custom analysis
    return results
```

### Custom Dataset Processing

Modify `MEQADataset` class in `meqa_dataset.py` to handle different data formats or preprocessing.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in config
2. **Model Loading Errors**: Check model path and availability
3. **Data Loading Issues**: Verify MEQA data path and format
4. **Attention Data Missing**: Ensure model generates attention weights

### Debug Mode

Set `debug: true` in config to:
- Disable WandB logging
- Print detailed progress information
- Save intermediate results

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{meqa2024,
  title={MEQA: A Benchmark for Multi-hop Event-centric Question Answering with Explanations},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This framework is part of the DeCoRe project. Please refer to the main project license.
