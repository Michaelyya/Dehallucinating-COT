#!/usr/bin/env python3
"""
Efficient Reasoning Head Discovery Script

This script discovers reasoning heads using efficient methods that provide
100-1000x speedup over brute-force ablation.

=============================================================================
EFFICIENT METHODS (NEW)
=============================================================================

1. "activation" - Activation-based scoring
   - Single forward pass per example
   - Scores based on activation norm and variance
   - Speedup: ~1000x
   - Best for: Quick screening, large models

2. "gradient" - Gradient-based scoring
   - Forward + backward pass per example
   - Scores based on gradient magnitude
   - Speedup: ~500x
   - Best for: Higher accuracy when answers are available

3. "attention_pattern" - Attention pattern analysis
   - Single forward pass with attention outputs
   - Scores based on entropy and focus
   - Speedup: ~1000x
   - Best for: Fastest possible screening

4. "combined" - Combined multi-method scoring
   - Uses all three methods with weighted combination
   - Speedup: ~300x
   - Best for: Highest accuracy, important downstream tasks

5. "ablation" - Original brute-force method (for comparison)
   - Full generation per head per example
   - Slowest but most direct measurement
   - Use only for small-scale validation

=============================================================================
USAGE
=============================================================================

# Fast discovery with activation-based scoring (RECOMMENDED)
python discover_reasoning_heads_efficient.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --method activation \
    --n_examples 50 \
    --top_k 20

# High-accuracy discovery with combined scoring
python discover_reasoning_heads_efficient.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --method combined \
    --n_examples 30 \
    --top_k 20

# Quick screening with attention patterns
python discover_reasoning_heads_efficient.py \
    --model_name "Qwen/Qwen3-4B-Instruct-2507" \
    --method attention_pattern \
    --n_examples 100 \
    --top_k 30

=============================================================================
OUTPUT
=============================================================================

Discovered heads are saved in DeCoReEntropy format:
    retrieval_heads/{model_name}_reasoning.json

Format: {"layer-head": [score], ...}
Example: {"5-12": [0.823], "7-3": [0.756], ...}

This format is directly compatible with DeCoReVanilla and DeCoReEntropy
for contrastive decoding.
"""

import os
import sys
import argparse
import json
import torch
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CACHE_DIR = "/cluster/scratch/yongyu/cache"


def load_model_and_tokenizer(model_name: str, cache_dir: str):
    """Load model and tokenizer, trying factory first then direct loading."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Set cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    print(f"Using cache directory: {cache_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load using factory (for models that support block_list)
    try:
        from src.factories import get_model
        from src.configs import ModelConfigs, DecoderConfigs, ModelConfig

        # Determine model type from name
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            model_type_name = "Qwen-Instruct"
        elif "llama" in model_name_lower:
            model_type_name = "LLaMA3-8b-Instruct"
        elif "mistral" in model_name_lower:
            model_type_name = "Mistral-Instruct"
        else:
            model_type_name = model_name.split("/")[-1] if "/" in model_name else model_name

        model_configs = ModelConfigs(
            name=model_type_name,
            model_type="instruct",
            configs=ModelConfig(
                model_name_or_path=model_name,
                max_seq_len=4096,
                max_new_tokens=100
            )
        )

        decoder_configs = DecoderConfigs(
            name="baseline",
            method="baseline",
            configs=ModelConfig()
        )

        model_wrapper = get_model(model_configs, decoder_configs)
        model = model_wrapper.model
        print(f"✓ Loaded model using factory")
    except Exception as e:
        print(f"Note: Could not load model using factory, using direct loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        ).eval()

    return model, tokenizer


def load_examples(dataset_path: str, n_examples: int) -> List[Dict[str, Any]]:
    """Load examples from dataset file."""
    examples = []

    if not os.path.exists(dataset_path):
        # Try to find dataset in common locations
        possible_paths = [
            dataset_path,
            os.path.join(os.path.dirname(__file__), dataset_path),
            os.path.join(os.path.dirname(__file__), "cognitive_mirrors_logical_reasoning.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "CognitiveMirrors", "dataset", "balanced_cot_train_data.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found. Tried:")
        print(f"  - {dataset_path}")
        print(f"\nPlease provide a valid dataset path or run:")
        print(f"  python preprocess_cognitive_mirrors.py")
        return []

    print(f"Loading examples from: {dataset_path}")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data[:n_examples * 2]:  # Load extra to account for parsing failures
            if isinstance(item, dict):
                # Try to extract relevant fields
                example = {}
                if "question" in item:
                    example["question"] = item["question"]
                if "subquestion" in item:
                    example["subquestion"] = item["subquestion"]
                if "answer" in item:
                    example["answer"] = item["answer"]
                if "subquestion_answer" in item:
                    example["subquestion_answer"] = item["subquestion_answer"]

                if example:
                    examples.append(example)

                if len(examples) >= n_examples:
                    break

    except json.JSONDecodeError:
        # Try line-by-line format
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n_examples:
                    break
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        examples.append(item)
                    except json.JSONDecodeError:
                        # Parse as backward-chaining format
                        if '|' in line:
                            parts = line.split('|')
                            edges_str = parts[0]
                            goal_path = parts[1].split(':')
                            edges = []
                            for edge in edges_str.split(','):
                                if '>' in edge:
                                    src, tgt = edge.split('>')
                                    edges.append([int(src), int(tgt)])
                            examples.append({
                                "edges": edges,
                                "goal": int(goal_path[0]) if goal_path[0].isdigit() else goal_path[0],
                                "path": [int(p) for p in goal_path[1].split('>') if p.isdigit()] if len(goal_path) > 1 else []
                            })

    print(f"Loaded {len(examples)} examples")
    return examples[:n_examples]


def save_heads_decore_format(
    heads: List,
    output_dir: str,
    model_name: str,
    suffix: str = "reasoning"
) -> str:
    """Save discovered heads in DeCoReEntropy format."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DeCoRe format: {"layer-head": [score], ...}
    decore_format = {}
    for head in heads:
        key = f"{head.layer}-{head.head}"
        decore_format[key] = [head.score]

    # Sort by score (descending)
    sorted_heads = sorted(decore_format.items(), key=lambda x: x[1][0], reverse=True)
    decore_format = dict(sorted_heads)

    # Extract model base name
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name

    # Save to file
    output_file = os.path.join(output_dir, f"{model_base}_{suffix}.json")
    with open(output_file, 'w') as f:
        f.write(json.dumps(decore_format))

    print(f"✓ Saved {len(decore_format)} heads to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Efficient Reasoning Head Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast activation-based discovery (RECOMMENDED)
  python discover_reasoning_heads_efficient.py --method activation --n_examples 50

  # High-accuracy combined scoring
  python discover_reasoning_heads_efficient.py --method combined --n_examples 30

  # Quick attention pattern screening
  python discover_reasoning_heads_efficient.py --method attention_pattern --n_examples 100

Efficiency comparison (for Qwen3-4B with 1152 heads, 20 examples):
  - ablation:         ~3.5M forward passes (baseline)
  - activation:       ~20 forward passes   (~1000x faster)
  - gradient:         ~40 fwd+bwd passes   (~500x faster)
  - attention_pattern: ~20 forward passes  (~1000x faster)
  - combined:         ~60 forward passes   (~300x faster)
        """
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name or path"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="activation",
        choices=["activation", "gradient", "attention_pattern", "combined", "ablation"],
        help="Scoring method (default: activation)"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=50,
        help="Number of examples for scoring (default: 50)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top heads to select (default: 20)"
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Maximum layers to score (default: all)"
    )
    parser.add_argument(
        "--max_heads_per_layer",
        type=int,
        default=None,
        help="Maximum heads per layer to score (default: all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cognitive_mirrors_logical_reasoning.json",
        help="Dataset file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../retrieval_heads/",
        help="Output directory for discovered heads"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for models (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--use_gradient",
        action="store_true",
        help="Enable gradient computation in combined scorer (slower but more accurate)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    args = parser.parse_args()

    # Print configuration
    print(f"\n{'='*80}")
    print("EFFICIENT REASONING HEAD DISCOVERY")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"Examples: {args.n_examples}")
    print(f"Top K: {args.top_k}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load examples
    examples = load_examples(args.dataset, args.n_examples)
    if len(examples) == 0:
        print("ERROR: No examples loaded. Exiting.")
        return

    # Create scorer
    print(f"\n{'='*80}")
    print(f"CREATING {args.method.upper()} SCORER")
    print(f"{'='*80}\n")

    if args.method == "ablation":
        # Use original ablation scorer for comparison
        from reasoning_heads.head_scoring import AblationScorer
        scorer = AblationScorer(model, tokenizer, device)
    else:
        # Use efficient scorer
        from reasoning_heads.efficient_head_scoring import create_efficient_scorer
        kwargs = {
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
        }
        if args.method == "combined":
            kwargs["use_gradient"] = args.use_gradient

        scorer = create_efficient_scorer(args.method, model, tokenizer, device, **kwargs)

    # Score all heads
    print(f"\n{'='*80}")
    print("SCORING HEADS")
    print(f"{'='*80}\n")

    head_scores = scorer.score_all_heads(
        examples,
        subtask_name="logical_reasoning",
        max_layers=args.max_layers,
        max_heads_per_layer=args.max_heads_per_layer
    )

    # Select top K
    top_heads = head_scores[:args.top_k]

    # Print results
    print(f"\n{'='*80}")
    print("DISCOVERED REASONING HEADS")
    print(f"{'='*80}\n")

    print(f"Top {args.top_k} reasoning heads by {args.method} score:\n")
    for i, head in enumerate(top_heads):
        print(f"  {i+1:2d}. Layer {head.layer:2d}, Head {head.head:2d}: "
              f"score={head.score:.6f}, confidence={head.confidence:.2f}")
        if head.metadata:
            meta_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in head.metadata.items()
                                  if k not in ["error", "weights"]])
            if meta_str:
                print(f"      {meta_str}")

    # Save in DeCoReEntropy format
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    # Resolve output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    output_file = save_heads_decore_format(
        top_heads,
        output_dir,
        args.model_name,
        suffix=f"reasoning_{args.method}"
    )

    # Also save with generic name for easy loading
    generic_file = save_heads_decore_format(
        top_heads,
        output_dir,
        args.model_name,
        suffix=""
    )

    print(f"\n{'='*80}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Discovered {len(top_heads)} reasoning heads using {args.method} method")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Generic file: {generic_file}")
    print(f"\nTo use with DeCoReVanilla/DeCoReEntropy:")
    print(f"  retrieval_heads_dir: '{os.path.dirname(output_file)}'")
    print(f"  num_retrieval_heads: {len(top_heads)}")


if __name__ == "__main__":
    main()
