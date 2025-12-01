#!/usr/bin/env python3
"""
Discover reasoning heads from Atomic Task datasets.

This script discovers attention heads important for different types of transitive reasoning
by measuring performance drops when individual heads are ablated.

Usage:
    python discover_atomic_task_heads.py --model_name "Qwen/Qwen3-4B-Instruct-2507"
    python discover_atomic_task_heads.py --task_type scalar-max --n_examples 20
    python discover_atomic_task_heads.py --all_tasks --top_k 20
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch
from tqdm import tqdm 

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
testing_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(testing_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, testing_dir)
sys.path.insert(0, script_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import head scoring utilities
from head_scoring import AblationScorer, HeadScore, ATOMIC_TASK_TYPES, ATOMIC_TASK_PROMPTS

# Default cache directory
# DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", "/cluster/scratch/yongyu/cache")
# DEFAULT_CACHE_DIR = '/home/zpoints/links/scratch/.cache/huggingface/'
DEFAULT_CACHE_DIR = '/home/zpoints/links/scratch/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'



# Atomic task dataset directory
ATOMIC_DATASET_DIR = os.path.join(project_root, "atomic_task_dataset", "dataset")


def load_atomic_task_dataset(task_name: str, n_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load examples from an atomic task dataset."""
    # Map task name to filename
    filename_map = {
        "scalar-max": "scalar-max.json",
        "symbolic-inequality": "symbolic-inequality.json",
        "temporal-order": "temporal-order.json",
        "spatial-containment": "spatial-containment.json",
        "subset-implication": "subset-implication.json",
        "hierarchy": "hierarchy.json",
    }
    
    if task_name not in filename_map:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(filename_map.keys())}")
    
    filepath = os.path.join(ATOMIC_DATASET_DIR, filename_map[task_name])
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {filepath}")
    
    if n_examples and n_examples < len(data):
        data = data[:n_examples]
        print(f"Using first {n_examples} examples")
    
    return data


def discover_heads_for_task(
    model,
    tokenizer,
    task_name: str,
    n_examples: int = 20,
    top_k: int = 20,
    device: str = "cuda",
    max_layers: int = None,
    max_heads_per_layer: int = None,
    debug: bool = False
) -> List[HeadScore]:
    """Discover important reasoning heads for a specific atomic task."""
    
    print(f"\n{'='*60}")
    print(f"Discovering heads for: {task_name}")
    print(f"{'='*60}")
    
    # Load dataset
    examples = load_atomic_task_dataset(task_name, n_examples)
    
    if len(examples) == 0:
        print(f"No examples found for task {task_name}")
        return []
    
    # Get the task type name for evaluation
    task_type = ATOMIC_TASK_TYPES.get(task_name, task_name)
    
    # Show which task-specific prompt is being used
    if task_type in ATOMIC_TASK_PROMPTS:
        print(f"Using task-specific prompt for: {task_type}")
        # Show first few lines of the prompt
        prompt_preview = ATOMIC_TASK_PROMPTS[task_type].split('\n')[:3]
        print(f"  Prompt preview: {' '.join(prompt_preview)[:100]}...")
    else:
        print(f"Using generic prompt (no task-specific prompt for: {task_type})")
    
    # Create scorer
    scorer = AblationScorer(model, tokenizer, device)
    
    # Get model dimensions
    n_layers = getattr(model.config, 'num_hidden_layers', 
                      getattr(model.config, 'n_layers', 32))
    n_heads = getattr(model.config, 'num_attention_heads',
                     getattr(model.config, 'n_heads', 32))
    
    # Set limits
    if max_layers is None:
        # max_layers = min(n_layers, 8)  # Default to first 8 layers
        max_layers = n_layers
    if max_heads_per_layer is None:
        # max_heads_per_layer = min(n_heads, 8)  # Default to first 8 heads
        max_heads_per_layer = n_heads
    
    print(f"Model has {n_layers} layers × {n_heads} heads")
    print(f"Scoring {max_layers} layers × {max_heads_per_layer} heads = {max_layers * max_heads_per_layer} heads")
    print(f"Using {len(examples)} examples for scoring")
    
    # Score all heads
    scores = scorer.score_all_heads(
        examples=examples,
        subtask_name=task_type,
        n_layers=n_layers,
        n_heads=n_heads,
        max_layers=max_layers,
        max_heads_per_layer=max_heads_per_layer
    )
    
    # Filter and sort by score (descending - largest performance drop)
    # Positive scores indicate important heads (baseline > ablated)
    positive_scores = [s for s in scores if s.score > 0]
    positive_scores.sort(key=lambda x: x.score, reverse=True)
    
    print(f"\nResults for {task_name}:")
    print(f"  Total heads scored: {len(scores)}")
    print(f"  Heads with positive score: {len(positive_scores)}")
    if positive_scores:
        print(f"  Score range: {positive_scores[-1].score:.4f} to {positive_scores[0].score:.4f}")
        print(f"  Top 5 heads:")
        for i, s in enumerate(positive_scores[:5]):
            print(f"    {i+1}. Layer {s.layer}, Head {s.head}: score={s.score:.4f}")
    
    # Return top K heads
    return positive_scores[:top_k]


def save_heads_for_task(
    heads: List[HeadScore],
    task_name: str,
    output_dir: str,
    model_name: str
) -> str:
    """Save discovered heads for a task in DeCoReEntropy format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DeCoRe format: {"layer-head": [score], ...}
    decore_format = {}
    for head in heads:
        key = f"{head.layer}-{head.head}"
        decore_format[key] = [head.score]
    
    # Sort by score descending
    sorted_heads = sorted(decore_format.items(), key=lambda x: x[1][0], reverse=True)
    decore_format = dict(sorted_heads)
    
    # Create filename based on model and task
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name
    filename = f"{model_base}_{task_name}.json"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        f.write(json.dumps(decore_format))
    
    print(f"Saved {len(decore_format)} heads to {output_path}")
    return output_path


def save_combined_heads(
    all_heads: Dict[str, List[HeadScore]],
    output_dir: str,
    model_name: str,
    top_k: int = 20
) -> str:
    """
    Combine heads from all tasks and save as a single file.
    Heads that appear in multiple tasks get higher priority.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count head occurrences across tasks and aggregate scores
    head_counts = {}  # (layer, head) -> count
    head_scores = {}  # (layer, head) -> list of scores
    
    for task_name, heads in all_heads.items():
        for head in heads:
            key = (head.layer, head.head)
            if key not in head_counts:
                head_counts[key] = 0
                head_scores[key] = []
            head_counts[key] += 1
            head_scores[key].append(head.score)
    
    # Calculate combined score: average score * sqrt(count) to reward consistency
    combined_scores = {}
    for key, scores in head_scores.items():
        avg_score = sum(scores) / len(scores)
        count_bonus = (head_counts[key] ** 0.5)  # sqrt to not over-weight
        combined_scores[key] = avg_score * count_bonus
    
    # Sort by combined score
    sorted_heads = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top K and convert to DeCoRe format
    decore_format = {}
    for (layer, head), score in sorted_heads[:top_k]:
        key = f"{layer}-{head}"
        decore_format[key] = [score]
    
    # Save combined file
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name
    filename = f"{model_base}_combined_reasoning.json"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        f.write(json.dumps(decore_format))
    
    print(f"\n{'='*60}")
    print(f"Combined heads from {len(all_heads)} tasks")
    print(f"{'='*60}")
    print(f"Saved top {len(decore_format)} combined reasoning heads to {output_path}")
    print(f"\nTop 10 combined reasoning heads:")
    for i, ((layer, head), score) in enumerate(sorted_heads[:10]):
        count = head_counts[(layer, head)]
        print(f"  {i+1}. Layer {layer}, Head {head}: combined_score={score:.4f} (in {count} tasks)")
    
    return output_path


def save_detailed_results(
    all_heads: Dict[str, List[HeadScore]],
    output_path: str,
    model_name: str
):
    """Save detailed results for analysis."""
    results = {
        "model": model_name,
        "tasks": {}
    }
    
    for task_name, heads in all_heads.items():
        results["tasks"][task_name] = {
            "n_heads": len(heads),
            "heads": [
                {
                    "layer": h.layer,
                    "head": h.head,
                    "score": h.score,
                    "confidence": h.confidence,
                    "metadata": h.metadata
                }
                for h in heads
            ]
        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover reasoning heads from Atomic Task datasets"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name or path"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=list(ATOMIC_TASK_TYPES.keys()),
        help="Specific task type to run (if not specified, runs all tasks)"
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="Run discovery on all atomic tasks"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=20,
        # default=1,
        help="Number of examples per task for scoring"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        # default=5,
        default=1000000,
        help="Top K heads to select per task"
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        # default=5,
        help="Maximum layers to scan (default: min(8, total_layers))"
    )
    parser.add_argument(
        "--max_heads_per_layer",
        type=int,
        default=None,
        # default=5,
        help="Maximum heads per layer to scan (default: min(8, total_heads))"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/home/zpoints/links/scratch/deha/hi',
        # default='/home/zpoints/links/scratch/deha/im',
        help="Output directory for results (default: retrieval_heads/)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for models (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "retrieval_heads")
    
    # Set cache directory
    # os.environ["HF_HOME"] = args.cache_dir
    # os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    # os.makedirs(args.cache_dir, exist_ok=True)
    # print(f"Using cache directory: {args.cache_dir}")
    
    # Determine which tasks to run
    if args.task_type:
        tasks_to_run = [args.task_type]
    elif args.all_tasks:
        tasks_to_run = list(ATOMIC_TASK_TYPES.keys())
    else:
        # Default: run all tasks
        tasks_to_run = list(ATOMIC_TASK_TYPES.keys())
    
    print(f"\n{'='*80}")
    print("ATOMIC TASK REASONING HEAD DISCOVERY")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Tasks: {tasks_to_run}")
    print(f"Examples per task: {args.n_examples}")
    print(f"Top K heads per task: {args.top_k}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model and tokenizer
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        # args.model_name,
        DEFAULT_CACHE_DIR,
        local_files_only=True,
        # cache_dir=args.cache_dir,
        # trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        # args.model_name,
        DEFAULT_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        # cache_dir=args.cache_dir,
        # trust_remote_code=True
    ).eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}")
    
    # Run discovery for each task
    all_heads = {}
    
    for task_name in tasks_to_run:
        try:
            heads = discover_heads_for_task(
                model=model,
                tokenizer=tokenizer,
                task_name=task_name,
                n_examples=args.n_examples,
                top_k=args.top_k,
                device=device,
                max_layers=args.max_layers,
                max_heads_per_layer=args.max_heads_per_layer,
                debug=args.debug
            )
            
            all_heads[task_name] = heads
            
            # Save individual task results
            if heads:
                save_heads_for_task(
                    heads=heads,
                    task_name=task_name,
                    output_dir=args.output_dir,
                    model_name=args.model_name
                )
        
        except Exception as e:
            print(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results if multiple tasks
    if len(all_heads) > 1:
        save_combined_heads(
            all_heads=all_heads,
            output_dir=args.output_dir,
            model_name=args.model_name,
            top_k=args.top_k
        )
    
    # Save detailed results
    model_base = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
    detailed_path = os.path.join(args.output_dir, f"{model_base}_atomic_tasks_detailed.json")
    save_detailed_results(all_heads, detailed_path, args.model_name)
    
    print(f"\n{'='*80}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(all_heads)} tasks")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nTo use these heads with DeCoReEntropy, set:")
    print(f"  retrieval_heads_dir: '{args.output_dir}'")
    print(f"  For individual task: use {model_base}_<task>.json")
    print(f"  For combined reasoning: use {model_base}_combined_reasoning.json")


if __name__ == "__main__":
    main()

