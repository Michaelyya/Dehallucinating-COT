# Reasoning Head Discovery Framework

A reproducible framework for identifying and evaluating reasoning heads used by models for backward-chaining style reasoning. This framework integrates with the existing testing benchmarks (HotpotQA, MEQA, MuSiQue) to evaluate whether the DeCoRe method uses those reasoning heads and whether masking them changes benchmark performance.

## Overview

This framework:

1. **Discovers subtasks** automatically from backward-chaining-circuits dataset and code
2. **Identifies reasoning heads** using multiple scoring methods (ablation, causal patching, mutual information)
3. **Evaluates head masking** on benchmark performance
4. **Generates comprehensive reports** with metrics, effect sizes, and statistical significance

## Installation

The framework uses the existing dependencies from the main project. Ensure you have:

- PyTorch
- Transformers
- NumPy, SciPy
- NetworkX (for graph analysis)
- scikit-learn (for mutual information)

## Quick Start

### 1. Discover Reasoning Heads

```bash
cd testing/reasoning_heads
python main.py --discover \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --backward_chaining_dir "../../backward-chaining-circuits" \
    --scoring_method ablation \
    --n_examples 20 \
    --top_k 10 \
    --output_heads discovered_heads.json
```

### 2. Evaluate on Benchmarks

```bash
python main.py --evaluate \
    --heads_file discovered_heads.json \
    --top_k_heads 10 \
    --output_results evaluation_results.json \
    --output_report evaluation_report.md
```

### 3. Full Pipeline

```bash
python main.py --discover --evaluate \
    --config config.yaml
```

## Architecture

### Components

1. **`subtask_extraction.py`**: Discovers subtasks from backward-chaining-circuits
   - Parses dataset examples
   - Analyzes source code
   - Identifies reasoning subtasks (edge parsing, path finding, goal identification, etc.)

2. **`head_scoring.py`**: Implements head scoring methods
   - `AblationScorer`: Measures effect of zeroing/randomizing heads
   - `CausalPatchingScorer`: Uses causal attention patching
   - `MutualInfoScorer`: Calculates mutual information between activations and labels

3. **`discovery.py`**: Main discovery framework
   - Orchestrates subtask discovery and head scoring
   - Collects head traces (attention maps, activations)
   - Saves discovered heads

4. **`evaluation.py`**: Benchmark evaluation
   - Runs benchmarks with and without head masking
   - Compares baseline vs masked performance
   - Calculates effect sizes and statistical significance

5. **`reporting.py`**: Report generation
   - Creates markdown/HTML reports
   - Includes metrics, head lists, effect sizes
   - Provides statistical analysis

## Configuration

See `config.yaml` for configuration options:

- **Model**: Model name and device
- **Discovery**: Scoring method, number of examples, thresholds
- **Evaluation**: Benchmark configs, filtering options
- **Output**: File paths for results and reports

## Subtasks

The framework automatically discovers subtasks such as:

- **edge_parsing**: Parse edge tokens and construct graph
- **goal_identification**: Identify goal node from input
- **path_finding**: Find path from root to goal using backward-chaining
- **node_traversal**: Traverse graph nodes step-by-step
- **graph_construction**: Construct graph representation
- **backward_chain_step**: Execute one reasoning step
- **path_validation**: Validate generated path
- **token_prediction**: Predict next token in sequence

## Scoring Methods

### Ablation
Measures performance drop when head is zeroed or randomized. Higher score = more important head.

### Causal Patching
Replaces head activations with baseline and measures effect on output logits.

### Mutual Information
Calculates mutual information between head activations and subtask labels.

## Output Format

### Discovered Heads (`discovered_heads.json`)
```json
{
  "model": "llama",
  "scoring_method": "ablation",
  "n_subtasks": 8,
  "heads": [
    {
      "layer": 5,
      "head": 12,
      "subtask": "path_finding",
      "score": 0.85,
      "confidence": 0.92,
      "method": "ablation",
      "metadata": {...}
    }
  ]
}
```

### Evaluation Results (`evaluation_results.json`)
```json
{
  "hotpotqa": {
    "baseline": {...},
    "masked": {...},
    "comparison": {
      "metrics_comparison": {...},
      "effect_sizes": {...},
      "statistical_tests": {...}
    }
  }
}
```

### Report (`evaluation_report.md`)
Comprehensive markdown report with:
- Executive summary
- Per-benchmark metrics comparison
- Effect sizes (Cohen's d)
- Statistical significance tests
- Discovered heads summary
- Methodology

## Integration with DeCoRe

The framework integrates with DeCoRe's head masking mechanism:

1. Discovered heads are formatted as `(layer, head)` tuples
2. These are passed to DeCoRe's `block_list` parameter
3. DeCoRe masks these heads during generation
4. Performance is compared with and without masking

### Implementing Head Masking

To properly mask reasoning heads in the benchmarks, you have two options:

#### Option 1: Modify Existing Decoder (Recommended for Testing)

Modify the test scripts to pass `block_list` parameter when calling model.generate():

```python
# In test_hotpotqa.py, test_meqa.py, or test_musique.py
masked_heads = config.get("masked_heads", [])
if masked_heads:
    # Modify model calls to include block_list
    outputs = model.model(
        input_ids=inputs,
        block_list=masked_heads,  # Add this parameter
        ...
    )
```

#### Option 2: Create Custom Decoder Class

Create a new decoder class that extends BaseModel and masks reasoning heads:

```python
# In src/models/reasoning_head_masked.py
from src.models.base_model import BaseModel

class ReasoningHeadMaskedModel(BaseModel):
    def __init__(self, model_configs, decoder_configs):
        super().__init__(model_configs, decoder_configs)
        self.masked_heads = decoder_configs.configs.get("masked_heads", [])
    
    def generate(self, inputs, return_attentions=False):
        # Use block_list to mask heads during generation
        outputs = self.model(
            input_ids=inputs,
            block_list=self.masked_heads,
            ...
        )
        ...
```

Then register it in `src/factories.py` and use it in config files.

## Example Workflow

```python
from reasoning_heads import ReasoningHeadDiscovery, ReasoningHeadEvaluator

# Initialize discovery
discovery = ReasoningHeadDiscovery(
    model=model,
    tokenizer=tokenizer,
    backward_chaining_dir="../backward-chaining-circuits"
)

# Discover heads
heads = discovery.discover_heads(n_examples_per_subtask=20, top_k=10)
discovery.save_discovered_heads(heads, "heads.json")

# Evaluate
evaluator = ReasoningHeadEvaluator(discovery)
results = evaluator.evaluate_all_benchmarks(
    discovered_heads=heads,
    benchmark_configs={...}
)

# Generate report
from reasoning_heads import generate_evaluation_report
report = generate_evaluation_report(results, heads, "report.md")
```

## Troubleshooting

### Model Loading Issues
- Ensure model path is correct
- Check device availability (CUDA/CPU)
- Verify model supports attention extraction

### Discovery Issues
- Check backward-chaining-circuits directory path
- Ensure dataset file exists
- Verify sufficient examples in dataset

### Evaluation Issues
- Check benchmark config paths
- Ensure test scripts are in path
- Verify model supports block_list parameter

## Citation

If you use this framework, please cite:

```bibtex
@misc{reasoning_heads_framework,
  title={Reasoning Head Discovery Framework for Backward-Chaining Reasoning},
  author={...},
  year={2024}
}
```

## License

Part of the DeCoRe project. See main project license.

