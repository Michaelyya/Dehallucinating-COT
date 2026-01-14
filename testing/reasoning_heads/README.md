# Reasoning Head Discovery Framework

This framework identifies attention heads responsible for reasoning in transformer models. Discovered heads are used for **contrastive decoding** to reduce hallucinations and improve reasoning faithfulness.

## NEW: Efficient Discovery Methods (~100-1000x Faster)

The original ablation-based discovery is computationally expensive. We now provide **efficient alternatives** that achieve similar results with dramatically reduced compute:

| Method | Speedup | Description | Best For |
|--------|---------|-------------|----------|
| `activation` | ~1000x | Scores based on activation norm/variance | Quick screening, large models |
| `gradient` | ~500x | Scores based on gradient magnitude | High accuracy with answers available |
| `attention_pattern` | ~1000x | Analyzes attention entropy and focus | Fastest possible screening |
| `combined` | ~300x | Weighted combination of all methods | Maximum accuracy |
| `ablation` | 1x (baseline) | Original brute-force method | Ground truth comparison |

### Quick Start with Efficient Methods

```bash
cd Dehallucinating-COT
source Deha/bin/activate

# RECOMMENDED: Fast activation-based discovery
python testing/reasoning_heads/discover_reasoning_heads_efficient.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --method activation \
    --n_examples 50 \
    --top_k 20

# High-accuracy combined scoring
python testing/reasoning_heads/discover_reasoning_heads_efficient.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --method combined \
    --n_examples 30 \
    --top_k 20

# Quick attention pattern screening (fastest)
python testing/reasoning_heads/discover_reasoning_heads_efficient.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --method attention_pattern \
    --n_examples 100 \
    --top_k 30
```

### Efficient Methods Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--method` | Scoring method (`activation`, `gradient`, `attention_pattern`, `combined`, `ablation`) | `activation` |
| `--model_name` | HuggingFace model name or path | `Qwen/Qwen3-4B-Instruct-2507` |
| `--n_examples` | Number of examples for scoring | `50` |
| `--top_k` | Top K heads to select | `20` |
| `--max_layers` | Maximum layers to scan (default: all) | `None` |
| `--max_heads_per_layer` | Maximum heads per layer (default: all) | `None` |
| `--dataset` | Dataset file path | `cognitive_mirrors_logical_reasoning.json` |
| `--output_dir` | Output directory for results | `retrieval_heads/` |
| `--use_gradient` | Enable gradient in combined scorer | `False` |
| `--batch_size` | Batch size for processing | `1` |
| `--max_seq_len` | Maximum sequence length | `512` |

### Programmatic Usage

```python
from reasoning_heads import create_efficient_scorer, create_scorer

# Method 1: Direct efficient scorer
from reasoning_heads.efficient_head_scoring import ActivationScorer
scorer = ActivationScorer(model, tokenizer, device="cuda")
scores = scorer.score_all_heads(examples, subtask_name="logical_reasoning")

# Method 2: Factory function (auto-routes to efficient methods)
scorer = create_scorer("activation", model, tokenizer, device)
scores = scorer.score_all_heads(examples, "logical_reasoning")

# Method 3: ReasoningHeadDiscovery class
from reasoning_heads import ReasoningHeadDiscovery
discovery = ReasoningHeadDiscovery(
    model, tokenizer,
    scoring_method="activation",  # Use efficient method
    device="cuda"
)
heads = discovery.discover_heads(n_examples_per_subtask=50, top_k=20)
```

---

## How Efficient Methods Work

### 1. Activation-Based Scoring (`activation`)

**Concept:** Important heads have larger activation outputs and higher variance across tokens.

```
For each example:
    1. Single forward pass with hooks on all attention layers
    2. Collect all head activations: [batch, seq, n_heads, head_dim]
    3. Compute per-head metrics:
       - Activation norm (L2 norm averaged over tokens)
       - Activation variance (variance across sequence positions)
    4. Score = weighted_sum(norm, variance)
```

**Why it works:** Heads with larger activation norms have more influence on final outputs. High variance indicates context-sensitive computation (reasoning) rather than static patterns (memorization).

### 2. Gradient-Based Scoring (`gradient`)

**Concept:** Heads with larger gradients have more causal influence on predictions.

```
For each example with answer:
    1. Forward pass tracking all head outputs
    2. Compute loss on target answer tokens
    3. Backward pass to get gradients
    4. Score = gradient magnitude for each head
```

**Why it works:** Gradient magnitude indicates how much changing a head's output would affect the loss. This directly measures causal importance.

### 3. Attention Pattern Analysis (`attention_pattern`)

**Concept:** Reasoning heads have focused (low-entropy) attention patterns.

```
For each example:
    1. Forward pass with output_attentions=True
    2. For each head, compute:
       - Attention entropy: -(attn * log(attn)).sum()
       - Max attention: max(attn)
    3. Score = (1 - normalized_entropy) + normalized_max_attn
```

**Why it works:** Reasoning heads attend decisively to relevant context tokens, while less important heads have diffuse attention.

### 4. Combined Scoring (`combined`)

Runs all three methods and combines scores with configurable weights:

```python
combined_score = (
    0.4 * normalized_activation_score +
    0.4 * normalized_gradient_score +  # if use_gradient=True
    0.2 * normalized_attention_score
)
```

---

## Original Ablation-Based Discovery

The original brute-force method is still available for comparison or when maximum accuracy is needed:

### Atomic Tasks Discovery

```bash
cd Dehallucinating-COT
source Deha/bin/activate
pip install -r requirements.txt

# Run discovery on all tasks (SLOW - uses ablation)
python testing/reasoning_heads/discover_atomic_task_heads.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --all_tasks \
    --n_examples 10 \
    --top_k 20 \
    --max_layers 8 \
    --max_heads_per_layer 8
```

### Command Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name or path | `Qwen/Qwen3-4B-Instruct-2507` |
| `--all_tasks` | Run discovery on all 6 atomic tasks | - |
| `--task_type` | Run on a single task (see below) | - |
| `--n_examples` | Number of examples per task for scoring | `20` |
| `--top_k` | Top K heads to select per task | `20` |
| `--max_layers` | Maximum layers to scan | `8` |
| `--max_heads_per_layer` | Maximum heads per layer to scan | `8` |
| `--output_dir` | Output directory for results | `retrieval_heads/` |
| `--cache_dir` | Model cache directory | `/cluster/scratch/yongyu/cache` |
| `--debug` | Enable debug output | `False` |

### Task Types

| Task Name | Description | Example Question |
|-----------|-------------|------------------|
| `scalar-max` | Find max/min from comparisons | "Alice is taller than Bob. Who is tallest?" |
| `symbolic-inequality` | Determine relation (>, <, =, unknown) | "A > B, B > C. What is A vs C?" |
| `temporal-order` | Find earliest/latest event | "Meeting before lunch. Which is first?" |
| `spatial-containment` | Find extreme position/container | "X is north of Y. Which is southernmost?" |
| `subset-implication` | Logical entailment (Yes/No) | "All X are Y. Is every X a Z?" |
| `hierarchy` | Find top-level in hierarchy | "A manages B. Who is top manager?" |

### Run Single Task

```bash
# Scalar comparisons only
python testing/reasoning_heads/discover_atomic_task_heads.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --task_type scalar-max \
    --n_examples 20

# Symbolic inequality only
python testing/reasoning_heads/discover_atomic_task_heads.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --task_type symbolic-inequality \
    --n_examples 20

# Hierarchy reasoning only
python testing/reasoning_heads/discover_atomic_task_heads.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --task_type hierarchy \
    --n_examples 20
```

---

## Output Format

All discovery methods output heads in **DeCoReEntropy format** for direct use with contrastive decoding:

```json
{
  "5-12": [0.823],
  "7-3": [0.756],
  "3-8": [0.698],
  ...
}
```

Format: `"layer-head": [score]`

### Output Files

Results are saved to `retrieval_heads/`:

| File | Description |
|------|-------------|
| `<model>_reasoning_<method>.json` | Heads discovered with specific method |
| `<model>.json` | Generic file for easy loading |
| `<model>_<task>.json` | Heads for individual task (atomic tasks) |
| `<model>_combined_reasoning.json` | Combined top heads across all tasks |
| `<model>_meqa.json` | Heads for MEQA dataset |

---

## How Ablation Discovery Works (Original Method)

1. **Baseline**: Run all examples without ablation → get accuracy
2. **Per-head ablation**: For each head, zero it out and run examples → get ablated accuracy
3. **Score**: `baseline_accuracy - ablated_accuracy`
   - Positive score = head is important (ablation hurts performance)
   - Negative score = head may be harmful (ablation helps)
4. **Select top K** heads with highest positive scores

**Computational cost**: O(layers × heads × examples × tokens)

For Qwen3-4B with 36 layers × 32 heads = 1152 heads and 20 examples:
- ~3.5M forward passes with ablation
- ~20 forward passes with activation-based method (**~175,000x fewer**)

---

## MEQA Dataset Discovery

To discover reasoning heads using the MEQA (Multi-hop Event-centric Question Answering) train dataset:

```bash
# Discover heads using MEQA train dataset
python testing/reasoning_heads/discover_meqa_heads.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --n_examples 20 \
    --top_k 20 \
    --max_layers 8 \
    --max_heads_per_layer 8
```

### MEQA Command Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name or path | `Qwen/Qwen3-4B-Instruct-2507` |
| `--n_examples` | Number of examples from MEQA train for scoring | `20` |
| `--top_k` | Top K heads to select | `20` |
| `--max_layers` | Maximum layers to scan | `8` |
| `--max_heads_per_layer` | Maximum heads per layer to scan | `8` |
| `--output_dir` | Output directory for results | `retrieval_heads/` |
| `--cache_dir` | Model cache directory | `/cluster/scratch/yongyu/cache` |
| `--debug` | Enable debug output | `False` |

### MEQA Output Files

Results are saved to `retrieval_heads/`:

| File | Description |
|------|-------------|
| `<model>_meqa.json` | **Top K reasoning heads for MEQA** (DeCoReEntropy format) |
| `<model>_meqa_detailed.json` | Detailed results with metadata |

---

## Integration with Contrastive Decoding

Discovered heads are used in DeCoReVanilla for training-free contrastive decoding:

```python
# In configs/decoder/decore_vanilla.yaml
name: DeCoReVanilla
method: DeCoReVanilla
configs:
  retrieval_heads_dir: ./retrieval_heads/
  num_retrieval_heads: 20
  alpha: 0.5
```

The contrastive decoding formula:
```
next_token_logits = (1 + alpha) * base_logits - alpha * hallucinated_logits
```

Where `hallucinated_logits` come from the model with discovered reasoning heads ablated.

---

## File Structure

```
testing/reasoning_heads/
├── __init__.py                           # Package exports
├── README.md                             # This file
├── config.yaml                           # Configuration
├── discovery.py                          # ReasoningHeadDiscovery class
├── head_scoring.py                       # Original ablation scorer
├── efficient_head_scoring.py             # NEW: Efficient scorers
├── discover_reasoning_heads.py           # Original discovery script
├── discover_reasoning_heads_efficient.py # NEW: Efficient discovery script
├── discover_atomic_task_heads.py         # Atomic task discovery
├── discover_meqa_heads.py                # MEQA discovery
├── evaluation.py                         # Evaluation utilities
├── reporting.py                          # Reporting utilities
├── preprocess_cognitive_mirrors.py       # Dataset preprocessing
└── test.py                               # Tests
```

---

## Research Basis

The efficient scoring methods are based on established interpretability research:

- **Activation importance**: Michel et al. (2019) "Are Sixteen Heads Really Better than One?"
- **Gradient-based attribution**: Voita et al. (2019) "Analyzing Multi-Head Self-Attention"
- **Attention patterns**: Elhage et al. (2022) "A Mathematical Framework for Transformer Circuits"

The core hypothesis: Certain attention heads are responsible for multi-hop reasoning. Masking these heads creates a "hallucinated" distribution, and contrasting with this distribution reduces hallucinations.
