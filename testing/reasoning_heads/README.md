
## Quick Start

```bash
cd Dehallucinating-COT
source Deha/bin/activate
pip install -r requirements.txt

# Run discovery on all tasks
python testing/reasoning_heads/discover_atomic_task_heads.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --all_tasks \
    --n_examples 10 \
    --top_k 20 \
    --max_layers 8 \
    --max_heads_per_layer 8
```

## Command Arguments

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

## Task Types

| Task Name | Description | Example Question |
|-----------|-------------|------------------|
| `scalar-max` | Find max/min from comparisons | "Alice is taller than Bob. Who is tallest?" |
| `symbolic-inequality` | Determine relation (>, <, =, unknown) | "A > B, B > C. What is A vs C?" |
| `temporal-order` | Find earliest/latest event | "Meeting before lunch. Which is first?" |
| `spatial-containment` | Find extreme position/container | "X is north of Y. Which is southernmost?" |
| `subset-implication` | Logical entailment (Yes/No) | "All X are Y. Is every X a Z?" |
| `hierarchy` | Find top-level in hierarchy | "A manages B. Who is top manager?" |

## Run Single Task

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

## Output Files

Results are saved to `retrieval_heads/`:

| File | Description |
|------|-------------|
| `<model>_<task>.json` | Heads for individual task |
| `<model>_combined_reasoning.json` | **Combined top 20 heads across all tasks** |
| `<model>_atomic_tasks_detailed.json` | Detailed results with metadata |

## How It Works

1. **Baseline**: Run all examples without ablation → get accuracy
2. **Per-head ablation**: For each head, zero it out and run examples → get ablated accuracy
3. **Score**: `baseline_accuracy - ablated_accuracy`
   - Positive score = head is important (ablation hurts performance)
   - Negative score = head may be harmful (ablation helps)
4. **Select top K** heads with highest positive scores


