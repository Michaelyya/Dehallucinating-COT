import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch
import numpy as np

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
testing_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(testing_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, testing_dir)
sys.path.insert(0, script_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import head scoring utilities
from head_scoring import AblationScorer, HeadScore

# Default cache directory
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", "/cluster/scratch/yongyu/cache")

# MEQA dataset directory
MEQA_DATASET_DIR = os.path.join(project_root, "data", "MEQA", "data")


def load_meqa_dataset(n_examples: Optional[int] = None, split: str = "train") -> List[Dict[str, Any]]:
    """Load examples from MEQA train dataset."""
    if split == "train":
        filename = "collected_train.json"
    elif split == "dev":
        filename = "collected_dev.json"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'dev'")
    
    filepath = os.path.join(MEQA_DATASET_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MEQA dataset file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {filepath}")
    
    if n_examples and n_examples < len(data):
        data = data[:n_examples]
        print(f"Using first {n_examples} examples")
    
    return data


def format_meqa_example(example: Dict[str, Any], tokenizer) -> str:
    """Format MEQA example for model input using chat template."""
    context = example.get("context", "")
    question = example.get("question", "")
    
    # Use the same format as MEQADataset
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. Follow the output schema exactly. "
                    "Use only evidence from the provided Context (no external knowledge). "
                    "Keep the final answer to a few words. In Explanation, keep each line concise."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context: {context}\n\n"
                    f"Question: {question}\n\n"
                    "First, provide your answer in a few words only (no prefix, just the answer).\n"
                    "Then, provide an explanation in this format:\n"
                    "Explanation:\n"
                    "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
                    "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
                    "3. Reasoning: <short 1-sentence link from evidence to the answer>"
                )
            }
        ]
        
        prompted_question = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,  # Don't tokenize, return text
        )
    else:
        # Simple format for base models
        prompted_question = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "First, provide your answer in a few words only (no prefix, just the answer).\n"
            "Then, provide an explanation in this format:\n"
            "Explanation:\n"
            "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
            "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
            "3. Reasoning: <short 1-sentence link from evidence to the answer>"
        )
    
    return prompted_question


class MEQAAblationScorer(AblationScorer):
    """Ablation scorer specifically for MEQA dataset."""
    
    def _simple_bleu(self, reference_tokens, candidate_tokens, n: int = 4) -> float:
        """
        Lightweight BLEU implementation (copied from head_scoring.simple_bleu)
        to compare baseline vs ablated generations on MEQA.
        """
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        # Brevity penalty
        if len(candidate_tokens) < len(reference_tokens):
            bp = np.exp(1 - len(reference_tokens) / len(candidate_tokens))
        else:
            bp = 1.0
        
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = {}
            for j in range(len(reference_tokens) - i + 1):
                ngram = tuple(reference_tokens[j:j+i])
                ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
            
            cand_ngrams = {}
            for j in range(len(candidate_tokens) - i + 1):
                ngram = tuple(candidate_tokens[j:j+i])
                cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
            
            matches = 0
            total = 0
            for ngram, count in cand_ngrams.items():
                total += count
                if ngram in ref_ngrams:
                    matches += min(count, ref_ngrams[ngram])
            
            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append(matches / total)
        
        if any(p == 0 for p in precisions):
            return 0.0
        
        geometric_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        return bp * geometric_mean
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format MEQA example for model input."""
        return format_meqa_example(example, self.tokenizer)
    
    def _evaluate_subtask(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        ablated_heads: Optional[List[tuple]] = None,
        debug: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate MEQA examples using both exact-match accuracy and BLEU.
        
        Returns:
            {
                "accuracy": exact_match_accuracy,
                "bleu": average_bleu,
                "correct": n_correct,
                "total": n_total,
            }
        """
        correct = 0
        total = 0
        bleu_scores: List[float] = []
        debug_printed = False
        
        for example in examples:
            try:
                # Format example using MEQA format
                input_text = format_meqa_example(example, self.tokenizer)
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
                
                # Generate with or without ablation
                with torch.no_grad():
                    if ablated_heads:
                        output = self._generate_with_ablation(input_ids, ablated_heads)
                    else:
                        output = self.model.generate(
                            input_ids,
                            max_new_tokens=150,
                            do_sample=False,
                            temperature=1.0,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                
                # Decode and extract generated text
                decoded_full = self.tokenizer.decode(output[0], skip_special_tokens=True)
                input_length = len(input_ids[0])
                generated_tokens = output[0][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Clean up generated text
                import re
                generated_text = re.sub(r'^assistant\s*\n*\s*', '', generated_text, flags=re.IGNORECASE)
                generated_text = generated_text.strip()
                
                if input_text in decoded_full:
                    generated_text_alt = decoded_full.replace(input_text, "", 1).strip()
                    generated_text_alt = re.sub(r'^assistant\s*\n*\s*', '', generated_text_alt, flags=re.IGNORECASE)
                    generated_text_alt = generated_text_alt.strip()
                    if len(generated_text_alt) > len(generated_text):
                        generated_text = generated_text_alt
                
                # Get correct answer
                correct_answer = example.get("answer", "")
                if isinstance(correct_answer, list):
                    correct_answer = ', '.join([str(a) for a in correct_answer if a is not None])
                else:
                    correct_answer = str(correct_answer)
                
                # Extract answer from generated text (first few words before "Explanation:")
                generated_answer = self._extract_meqa_answer(generated_text)
                
                # Check correctness (normalized comparison)
                is_correct = self._check_meqa_correctness(generated_answer, correct_answer)
                
                # BLEU between gold answer and generated answer
                ref_tokens = str(correct_answer).lower().split()
                cand_tokens = str(generated_answer).lower().split()
                bleu = self._simple_bleu(ref_tokens, cand_tokens)
                bleu_scores.append(bleu)
                
                # Optional debug print for the first example (both baseline and ablated)
                if debug and not debug_printed:
                    phase = "BASELINE" if not ablated_heads else "ABLATED"
                    print("\n---------- MEQA Debug Example ----------")
                    print(f"Phase     : {phase}")
                    print(f"Question  : {example.get('question', '')}")
                    print(f"Gold ans. : {correct_answer}")
                    print(f"Model out : {generated_text}")
                    print(f"Extracted : {generated_answer}")
                    print(f"BLEU      : {bleu:.4f}")
                    print("----------------------------------------\n")
                    debug_printed = True
                
                if is_correct:
                    correct += 1
                total += 1
                    
            except Exception as e:
                if debug:
                    print(f"Error processing example: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
        
        return {
            "accuracy": accuracy,
            "bleu": avg_bleu,
            "correct": correct,
            "total": total
        }
    
    def _extract_meqa_answer(self, generated_text: str) -> str:
        """Extract the answer from MEQA generation (before Explanation:)."""
        import re
        
        # Try to find answer before "Explanation:" marker
        explanation_match = re.search(r'Explanation\s*:', generated_text, re.IGNORECASE)
        if explanation_match:
            # Answer is everything before "Explanation:"
            answer_text = generated_text[:explanation_match.start()].strip()
        else:
            # No explanation marker, take first line or first sentence
            answer_text = generated_text.split('\n')[0].strip()
        
        # Clean up common prefixes
        answer_text = re.sub(r'^(answer|Answer)\s*:\s*', '', answer_text, flags=re.IGNORECASE)
        answer_text = answer_text.strip()
        
        # Take first few words (MEQA answers are typically short)
        words = answer_text.split()
        if len(words) > 10:
            answer_text = ' '.join(words[:10])
        
        return answer_text.strip()
    
    def _check_meqa_correctness(self, generated_answer: str, correct_answer: str) -> bool:
        """Check if generated answer matches correct answer (normalized comparison)."""
        if not generated_answer or not correct_answer:
            return False
        
        # Normalize both answers
        gen_norm = generated_answer.lower().strip()
        corr_norm = correct_answer.lower().strip()
        
        # Exact match
        if gen_norm == corr_norm:
            return True
        
        # Check if correct answer is contained in generated answer
        if corr_norm in gen_norm:
            return True
        
        # Check if generated answer is contained in correct answer (for partial matches)
        if gen_norm in corr_norm:
            return True
        
        # For list answers, check if any element matches
        if ',' in corr_norm:
            corr_parts = [p.strip() for p in corr_norm.split(',')]
            for part in corr_parts:
                if part in gen_norm or gen_norm in part:
                    return True
        
        # Token-level overlap (for fuzzy matching)
        gen_tokens = set(gen_norm.split())
        corr_tokens = set(corr_norm.split())
        
        if len(corr_tokens) > 0:
            overlap = len(gen_tokens & corr_tokens) / len(corr_tokens)
            if overlap >= 0.7:  # 70% token overlap
                return True
        
        return False


    def score_all_heads(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        max_layers: Optional[int] = None,
        max_heads_per_layer: Optional[int] = None
    ) -> List[HeadScore]:
        """
        Custom head scoring for MEQA that uses a combined accuracy + BLEU metric
        and prints detailed debug information for the first head.
        """
        if n_layers is None:
            n_layers = getattr(self.model.config, 'num_hidden_layers',
                               getattr(self.model.config, 'n_layers', 32))
        if n_heads is None:
            n_heads = getattr(self.model.config, 'num_attention_heads',
                              getattr(self.model.config, 'n_heads', 32))
        
        # Limit scope for faster initial discovery
        if max_layers is None:
            max_layers = min(n_layers, 8)
        if max_heads_per_layer is None:
            max_heads_per_layer = min(n_heads, 8)
        
        total_heads = max_layers * max_heads_per_layer
        
        # Baseline metrics (with debug to see one example)
        baseline_metrics = self._evaluate_subtask(
            examples, subtask_name, ablated_heads=None, debug=True
        )
        baseline_acc = baseline_metrics.get("accuracy", 0.0)
        baseline_bleu = baseline_metrics.get("bleu", 0.0)
        
        from tqdm import tqdm
        scores: List[HeadScore] = []
        first_head = True
        
        for layer in tqdm(range(max_layers), desc="  Layers", leave=False):
            for head in range(max_heads_per_layer):
                try:
                    debug = first_head  # print ablated debug only for first head
                    ablated_metrics = self._evaluate_subtask(
                        examples, subtask_name, ablated_heads=[(layer, head)], debug=debug
                    )
                    ablated_acc = ablated_metrics.get("accuracy", 0.0)
                    ablated_bleu = ablated_metrics.get("bleu", 0.0)
                    
                    # Combined score: drop in accuracy + drop in BLEU
                    acc_drop = baseline_acc - ablated_acc
                    bleu_drop = baseline_bleu - ablated_bleu
                    score_value = acc_drop + bleu_drop
                    
                    confidence = min(len(examples) / 10.0, 1.0)
                    
                    scores.append(
                        HeadScore(
                            layer=layer,
                            head=head,
                            score=score_value,
                            confidence=confidence,
                            method="ablation",
                            metadata={
                                "baseline_accuracy": baseline_acc,
                                "baseline_bleu": baseline_bleu,
                                "ablated_accuracy": ablated_acc,
                                "ablated_bleu": ablated_bleu,
                                "n_examples": len(examples),
                            },
                        )
                    )
                    first_head = False
                except Exception:
                    continue
        
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores

def discover_heads_for_meqa(
    model,
    tokenizer,
    n_examples: int = 20,
    top_k: int = 20,
    device: str = "cuda",
    max_layers: int = None,
    max_heads_per_layer: int = None,
    debug: bool = False
) -> List[HeadScore]:
    """Discover important reasoning heads for MEQA dataset."""
    
    print(f"\n{'='*60}")
    print(f"Discovering heads for: MEQA (Multi-hop Event-centric QA)")
    print(f"{'='*60}")
    
    # Load dataset
    examples = load_meqa_dataset(n_examples, split="train")
    
    if len(examples) == 0:
        print(f"No examples found for MEQA")
        return []
    
    # Create scorer (using MEQA-specific scorer)
    scorer = MEQAAblationScorer(model, tokenizer, device)
    
    # Get model dimensions
    n_layers = getattr(model.config, 'num_hidden_layers', 
                      getattr(model.config, 'n_layers', 32))
    n_heads = getattr(model.config, 'num_attention_heads',
                     getattr(model.config, 'n_heads', 32))
    
    # Set limits
    if max_layers is None:
        max_layers = min(n_layers, 8)  # Default to first 8 layers
    if max_heads_per_layer is None:
        max_heads_per_layer = min(n_heads, 8)  # Default to first 8 heads
    
    print(f"Model has {n_layers} layers × {n_heads} heads")
    print(f"Scoring {max_layers} layers × {max_heads_per_layer} heads = {max_layers * max_heads_per_layer} heads")
    print(f"Using {len(examples)} examples for scoring")
    
    # Score all heads
    scores = scorer.score_all_heads(
        examples=examples,
        subtask_name="meqa",  # Use "meqa" as subtask name
        n_layers=n_layers,
        n_heads=n_heads,
        max_layers=max_layers,
        max_heads_per_layer=max_heads_per_layer
    )
    
    # Filter and sort by score (descending - largest performance drop)
    # Positive scores indicate important heads (baseline > ablated)
    positive_scores = [s for s in scores if s.score > 0]
    positive_scores.sort(key=lambda x: x.score, reverse=True)
    
    print(f"\nResults for MEQA:")
    print(f"  Total heads scored: {len(scores)}")
    print(f"  Heads with positive score: {len(positive_scores)}")
    
    # Print global baseline metrics (same for all heads)
    if scores:
        any_meta = scores[0].metadata or {}
        baseline_acc = any_meta.get("baseline_accuracy", 0.0)
        baseline_bleu = any_meta.get("baseline_bleu", 0.0)
        print(f"  Baseline accuracy (no ablation): {baseline_acc:.4f}")
        print(f"  Baseline BLEU     (no ablation): {baseline_bleu:.4f}")
    
    # Print per-head ablation metrics for inspection
    print("\n  Per-head metrics (baseline vs ablated):")
    for s in scores:
        meta = s.metadata or {}
        b_acc = meta.get("baseline_accuracy", 0.0)
        b_bleu = meta.get("baseline_bleu", 0.0)
        a_acc = meta.get("ablated_accuracy", 0.0)
        a_bleu = meta.get("ablated_bleu", 0.0)
        print(
            f"    Layer {s.layer:2d}, Head {s.head:2d}: "
            f"score={s.score:+.4f} | "
            f"acc: {b_acc:.4f} -> {a_acc:.4f}, "
            f"bleu: {b_bleu:.4f} -> {a_bleu:.4f}"
        )
    
    if positive_scores:
        print(f"\n  Top 5 positive-score heads:")
        for i, s in enumerate(positive_scores[:5]):
            print(f"    {i+1}. Layer {s.layer}, Head {s.head}: score={s.score:.4f}")
    
    # Return top K heads
    return positive_scores[:top_k]


def save_heads_for_meqa(
    heads: List[HeadScore],
    output_dir: str,
    model_name: str
) -> str:
    """Save discovered heads for MEQA in DeCoReEntropy format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DeCoRe format: {"layer-head": [score], ...}
    decore_format = {}
    for head in heads:
        key = f"{head.layer}-{head.head}"
        decore_format[key] = [head.score]
    
    # Sort by score descending
    sorted_heads = sorted(decore_format.items(), key=lambda x: x[1][0], reverse=True)
    decore_format = dict(sorted_heads)
    
    # Create filename based on model
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name
    filename = f"{model_base}_meqa.json"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(decore_format, f, indent=2)
    
    print(f"Saved {len(decore_format)} heads to {output_path}")
    return output_path


def save_detailed_results(
    heads: List[HeadScore],
    output_path: str,
    model_name: str
):
    """Save detailed results for analysis."""
    results = {
        "model": model_name,
        "dataset": "MEQA",
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
        description="Discover reasoning heads from MEQA dataset"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name or path"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=20,
        help="Number of examples for scoring"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top K heads to select"
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Maximum layers to scan (default: min(8, total_layers))"
    )
    parser.add_argument(
        "--max_heads_per_layer",
        type=int,
        default=None,
        help="Maximum heads per layer to scan (default: min(8, total_heads))"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
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
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Using cache directory: {args.cache_dir}")
    
    print(f"\n{'='*80}")
    print("MEQA REASONING HEAD DISCOVERY")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: MEQA (train split)")
    print(f"Examples: {args.n_examples}")
    print(f"Top K heads: {args.top_k}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model and tokenizer
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True
    ).eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}")
    
    # Run discovery
    try:
        heads = discover_heads_for_meqa(
            model=model,
            tokenizer=tokenizer,
            n_examples=args.n_examples,
            top_k=args.top_k,
            device=device,
            max_layers=args.max_layers,
            max_heads_per_layer=args.max_heads_per_layer,
            debug=args.debug
        )
        
        if not heads:
            print("\nERROR: No reasoning heads discovered!")
            print("This might indicate:")
            print("  1. All heads have negative scores (masking improves performance)")
            print("  2. Baseline and ablated performance are too similar")
            print("  3. Model is not performing well on the task")
            return
        
        # Save results
        save_heads_for_meqa(
            heads=heads,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        
        # Save detailed results
        model_base = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
        detailed_path = os.path.join(args.output_dir, f"{model_base}_meqa_detailed.json")
        save_detailed_results(heads, detailed_path, args.model_name)
        
        print(f"\n{'='*80}")
        print("DISCOVERY COMPLETE")
        print(f"{'='*80}")
        print(f"Discovered {len(heads)} reasoning heads")
        print(f"Results saved to: {args.output_dir}")
        print(f"\nTo use these heads with DeCoReEntropy, set:")
        print(f"  retrieval_heads_dir: '{args.output_dir}'")
        print(f"  num_retrieval_heads: {len(heads)}")
        print(f"  Use file: {model_base}_meqa.json")
        
    except Exception as e:
        print(f"Error during discovery: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

