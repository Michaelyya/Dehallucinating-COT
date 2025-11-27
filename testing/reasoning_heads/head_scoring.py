
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy import stats
import json
import os
import re
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, leave=True):
        return iterable

# Atomic Task Types - maps folder names to task_type in JSON
ATOMIC_TASK_TYPES = {
    "scalar-max": "Transitive reasoning-scalar max",
    "symbolic-inequality": "Transitive reasoning-symbolic inequality", 
    "temporal-order": "Transitive reasoning-temporal order",
    "spatial-containment": "Transitive reasoning-spatial containment",
    "subset-implication": "Transitive reasoning-subset/implication",
    "hierarchy": "Transitive reasoning-hierarchy",
}

# Task-specific prompt templates
ATOMIC_TASK_PROMPTS = {
    "Transitive reasoning-scalar max": """You are an expert in transitive reasoning over scalar comparisons.

Task: Given comparative statements about entities (e.g., "A is taller than B", "B is older than C"), determine which entity is the maximum or minimum (tallest, shortest, oldest, youngest, heaviest, fastest, etc.).

Key reasoning skills required:
- Build a transitive ordering from comparative adjectives (taller/shorter, older/younger, heavier/lighter, faster/slower, richer/poorer, etc.)
- Handle interleaved comparisons (not just simple chains like A > B > C)
- Identify the unique maximum or minimum entity from the given constraints

Question:
{question}

Instructions:
- Read all comparative statements carefully
- Build the ordering by chaining comparisons transitively
- Identify which entity is at the extreme (maximum or minimum) as asked
- Your answer must be exactly one of the listed options

Output format:
[ "answer": "<your chosen option>" ]

Your answer:""",

    "Transitive reasoning-symbolic inequality": """You are an expert in transitive reasoning over symbolic inequalities.

Task: Given symbolic inequality statements (e.g., "A > B", "B > C", "D < A"), determine the relation between two specified variables from {">", "<", "=", "unknown"}.

Key reasoning skills required:
- Chain inequalities transitively (if A > B and B > C, then A > C)
- Handle criss-cross patterns and partially overlapping comparisons
- Recognize when the relation is genuinely underdetermined ("unknown")
- Handle equality cases correctly

Question:
{question}

Instructions:
- Parse all inequality statements
- Apply transitivity to derive the relation between the queried variables
- If the relation cannot be determined from given information, answer "unknown"
- Your answer must be exactly one of: ">", "<", "=", "unknown"

Output format:
[ "answer": "<your chosen option>" ]

Your answer:""",

    "Transitive reasoning-temporal order": """You are an expert in transitive reasoning over temporal relations.

Task: Given temporal relations between events (using "before"/"after", clock times, dates, or durations), determine which event happens first or last.

Key reasoning skills required:
- Parse temporal cues: "before"/"after", explicit times (9:00 AM, 3:00 PM), durations (30 minutes after), calendar references (Monday, Tuesday)
- Build a temporal ordering from mixed cues
- Handle interleaved temporal constraints
- Identify the unique earliest or latest event

Question:
{question}

Instructions:
- Read all temporal relations carefully
- Build the timeline by combining all temporal cues
- Identify which event is earliest (for "first") or latest (for "last")
- Your answer must be exactly one of the listed event names

Output format:
[ "answer": "<your chosen option>" ]

Your answer:""",

    "Transitive reasoning-spatial containment": """You are an expert in transitive reasoning over spatial relations and containment.

Task: Given spatial relations (left/right, north/south, above/below, inside/outside, distances) between objects or locations, determine the extreme position or ultimate container.

Key reasoning skills required:
- Parse directional relations (north of, left of, above, closer to)
- Handle numeric distances and coordinates
- Trace containment chains (X is in Y, Y is in Z → X is ultimately in Z)
- Identify the unique extreme position (leftmost, northernmost, highest, closest, outermost container)

Question:
{question}

Instructions:
- Read all spatial/containment relations
- Build the spatial ordering or containment hierarchy
- Identify the entity at the extreme position or the outermost container
- Your answer must be exactly one of the listed options

Output format:
[ "answer": "<your chosen option>" ]

Your answer:""",

    "Transitive reasoning-subset/implication": """You are an expert in transitive reasoning over class inclusion and logical implication.

Task: Given premises about categories and their relationships (e.g., "All X are Y", "No X are Y", "If X then Y"), determine whether a conclusion must be true, or select the correct conclusion.

Key reasoning skills required:
- Chain universal statements: "All X are Y" + "All Y are Z" → "All X are Z"
- Handle negations: "All X are Y" + "No Y are Z" → "No X are Z"
- Distinguish valid conclusions from plausible-sounding but incorrect ones
- Recognize when a conclusion does NOT follow (missing or reversed links)

Question:
{question}

Instructions:
- Parse all premise statements about category relationships
- Apply logical chaining to determine what must be true
- For Yes/No questions: answer "Yes" only if the conclusion is logically entailed
- For selection questions: choose the one statement that MUST be true
- Your answer must be exactly "Yes", "No", or the correct conclusion sentence

Output format:
[ "answer": "<your chosen option>" ]

Your answer:""",

    "Transitive reasoning-hierarchy": """You are an expert in transitive reasoning over hierarchical relations.

Task: Given hierarchical relations (manager/employee, parent/child, higher/lower rank, category/subcategory), determine ancestors, top-level managers, or the highest node in a hierarchy.

Key reasoning skills required:
- Chain hierarchical relations transitively (if A manages B and B manages C, then A is above C)
- Handle organizational hierarchies (manager, supervisor, director, CEO)
- Handle family trees (parent, grandparent, ancestor)
- Handle category taxonomies (subtype, supertype)
- Identify the unique answer for queries like "top-level manager", "grandparent", "highest-ranking person"

Question:
{question}

Instructions:
- Parse all hierarchical relations (manages, reports to, parent of, above in rank, etc.)
- Build the hierarchy tree
- Trace the path to answer queries about ancestors or top-level nodes
- Your answer must be exactly one of the listed options

Output format:
[ "answer": "<your chosen option>" ]

Your answer:"""
}

# Generic fallback prompt
ATOMIC_TASK_PROMPT_GENERIC = """You are an expert in multi-step logical and transitive reasoning. 
You will be given one multiple-choice question that already includes its answer options after the word "Options:". 
Use only the information in the question and no outside knowledge.

Question:
{question}

Instructions:
- Read the question and its options carefully.
- Treat the premises as the only ground truth; do not use any world knowledge or external facts.
- Use precise logical and transitive reasoning over the given information to determine which single option must be correct.
- Your chosen answer must be exactly one of the options as written in the question.
- Do not include your reasoning or any extra text.

Output format:
[ "answer": "<your chosen option, copied verbatim from the options>" ]

Your answer:"""

# BLEU score for evaluation
BLEU_AVAILABLE = False
USE_NLTK_BLEU = False
USE_EVALUATE_LIB = False
bleu_metric = None

def simple_bleu(reference_tokens, candidate_tokens, n=4):
    if len(candidate_tokens) == 0:
        return 0.0
    if len(reference_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    if len(candidate_tokens) < len(reference_tokens):
        if len(candidate_tokens) == 0:
            bp = 0.0
        else:
            bp = np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    else:
        bp = 1.0
    
    # N-gram precisions
    precisions = []
    for i in range(1, n + 1):
        # Get n-grams
        ref_ngrams = {}
        for j in range(len(reference_tokens) - i + 1):
            ngram = tuple(reference_tokens[j:j+i])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
        
        cand_ngrams = {}
        for j in range(len(candidate_tokens) - i + 1):
            ngram = tuple(candidate_tokens[j:j+i])
            cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
        
        # Count matches
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
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    geometric_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    return bp * geometric_mean

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
    USE_NLTK_BLEU = True
    USE_EVALUATE_LIB = False
except ImportError:
    try:
        # Try using evaluate library
        import evaluate
        bleu_metric = evaluate.load("bleu")
        BLEU_AVAILABLE = True
        USE_NLTK_BLEU = False
        USE_EVALUATE_LIB = True
    except ImportError:
        # Use simple fallback implementation
        BLEU_AVAILABLE = True
        USE_NLTK_BLEU = False
        USE_EVALUATE_LIB = False
        print("Note: Using simple BLEU implementation. For better accuracy, install nltk: pip install nltk")


@dataclass
class HeadScore:
    layer: int
    head: int
    score: float
    confidence: float
    method: str
    metadata: Dict[str, Any] = None
    
    def __repr__(self):
        return f"HeadScore(layer={self.layer}, head={self.head}, score={self.score:.4f}, confidence={self.confidence:.4f})"


class HeadScorer(ABC):
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @abstractmethod
    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str
    ) -> HeadScore:
        pass
    
    def score_all_heads(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        max_layers: Optional[int] = None,
        max_heads_per_layer: Optional[int] = None
    ) -> List[HeadScore]:
        if n_layers is None:
            n_layers = getattr(self.model.config, 'num_hidden_layers', 
                             getattr(self.model.config, 'n_layers', 32))
        if n_heads is None:
            n_heads = getattr(self.model.config, 'num_attention_heads',
                            getattr(self.model.config, 'n_heads', 32))
        
        # Limit scope for faster initial discovery
        if max_layers is None:
            max_layers = min(n_layers, 8)  # Limit to first 8 layers initially
        if max_heads_per_layer is None:
            max_heads_per_layer = min(n_heads, 8)  # Limit to first 8 heads per layer
        
        total_heads = max_layers * max_heads_per_layer
        print(f"  Scoring {total_heads} heads ({max_layers} layers × {max_heads_per_layer} heads per layer)")
        
        # Calculate baseline ONCE for all heads (more efficient and ensures consistency)
        print(f"  Calculating baseline performance (no heads ablated)...")
        baseline_metrics = self._evaluate_subtask(
            examples, subtask_name, ablated_heads=None, debug=False
        )
        baseline_acc = baseline_metrics.get("accuracy", 0)
        print(f"  Baseline BLEU: {baseline_acc:.4f}")
        
        scores = []
        from tqdm import tqdm
        
        # Debug mode for first head only
        first_head = True
        
        for layer in tqdm(range(max_layers), desc="  Layers", leave=False):
            for head in range(max_heads_per_layer):
                try:
                    # Only debug first head of first layer
                    debug = first_head and layer == 0 and head == 0
                    # Pass baseline to avoid recalculating
                    score = self.score_head(
                        layer, head, examples, subtask_name, 
                        baseline_acc=baseline_acc, debug=debug
                    )
                    scores.append(score)
                    first_head = False  # Turn off after first
                except Exception as e:
                    print(f"\n  Warning: Could not score layer {layer}, head {head}: {e}")
                    continue
        
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        print(f"  Completed scoring {len(scores)} heads")
        return scores


class AblationScorer(HeadScorer):
    
    def __init__(self, model, tokenizer, device: str = "cuda", ablation_type: str = "zero"):
        super().__init__(model, tokenizer, device)
        self.ablation_type = ablation_type  # "zero" or "random"
    
    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        baseline_acc: Optional[float] = None,
        debug: bool = False
    ) -> HeadScore:
        # Use all examples for more reliable scoring
        # This ensures we get meaningful differences between baseline and ablated
        scoring_examples = examples
        
        # Get baseline performance (reuse if provided, otherwise calculate)
        if baseline_acc is None:
            baseline_debug = debug
            baseline_metrics = self._evaluate_subtask(
                scoring_examples, subtask_name, ablated_heads=None, debug=baseline_debug
            )
            baseline_acc = baseline_metrics.get("accuracy", 0)
        else:
            # Use provided baseline (calculated once for all heads)
            baseline_acc = baseline_acc
        
        # Get performance with head ablated (also debug if in debug mode)
        # IMPORTANT: Make sure we're actually ablating the specific head
        if debug:
            print(f"\n    Ablating Layer {layer}, Head {head}...")
        
        ablated_metrics = self._evaluate_subtask(
            scoring_examples, subtask_name, ablated_heads=[(layer, head)], debug=debug
        )
        
        # Calculate score as absolute performance drop
        ablated_acc = ablated_metrics.get("accuracy", 0)
        
        # Score calculation: baseline - ablated
        # Positive score = head is important (masking hurts performance, baseline > ablated)
        # Negative score = head might be harmful (masking improves performance, ablated > baseline)
        # We want heads with LARGEST positive scores (most important for reasoning)
        score = baseline_acc - ablated_acc
        
        # Note: We keep negative scores as-is, but for reasoning head selection,
        # we focus on positive scores (heads that decrease performance when masked)
        
        # Confidence based on number of examples and consistency
        confidence = min(len(scoring_examples) / 10.0, 1.0)
        
        if debug:
            print(f"\n    Head scoring (Layer {layer}, Head {head}):")
            print(f"      Baseline accuracy: {baseline_acc:.4f}")
            print(f"      Ablated accuracy: {ablated_acc:.4f}")
            print(f"      Score: {score:.4f}")
            print(f"      Ablated heads used: [(layer={layer}, head={head})]")
        
        return HeadScore(
            layer=layer,
            head=head,
            score=score,
            confidence=confidence,
            method="ablation",
            metadata={
                "baseline_accuracy": baseline_acc,
                "ablated_accuracy": ablated_acc,
                "n_examples": len(scoring_examples)
            }
        )
    
    def _evaluate_subtask(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        ablated_heads: Optional[List[Tuple[int, int]]] = None,
        debug: bool = False
    ) -> Dict[str, float]:
        # Check if this is an atomic task (multiple-choice exact match)
        is_atomic_task = (subtask_name in ATOMIC_TASK_TYPES.values() or 
                         subtask_name in ATOMIC_TASK_TYPES.keys())
        if is_atomic_task:
            if debug:
                print(f"    Using atomic task evaluation (exact match accuracy) for: {subtask_name}")
            return self._evaluate_atomic_task(examples, subtask_name, ablated_heads, debug)
        
        # For CognitiveMirrors, use BLEU score instead of correctness
        use_bleu = subtask_name in ["logical_reasoning"]
        
        if use_bleu:
            return self._evaluate_subtask_bleu(examples, subtask_name, ablated_heads, debug)
        
        # Original correctness-based evaluation for backward-chaining
        correct = 0
        total = 0
        all_outputs = []  # For debugging
        
        for example in examples:
            try:
                # Convert example to input format
                input_text = self._format_example(example)
                
                # Debug: show formatted prompt for first example
                if debug and total == 0:
                    print(f"\n    DEBUG - Formatted prompt (first 500 chars):")
                    print(f"      {input_text[:500]}...")
                    if len(input_text) > 500:
                        print(f"      ... (total length: {len(input_text)} chars)")
                
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
                
                # Generate with or without ablation
                with torch.no_grad():
                    if ablated_heads:
                        output = self._generate_with_ablation(input_ids, ablated_heads)
                    else:
                        output = self.model.generate(
                            input_ids,
                            max_new_tokens=100,  # Increased for longer paths (can be 15+ nodes)
                            do_sample=False,
                            temperature=1.0,  # Not used when do_sample=False, but explicit
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                
                # Decode full output for debugging
                decoded_full = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the generated part (after the input)
                # The model generates: input + new_tokens
                # We need to extract only the new tokens
                input_length = len(input_ids[0])
                generated_tokens = output[0][input_length:]  # Only the newly generated tokens
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # For chat templates, the output might include "assistant" header
                # Strip common chat template artifacts
                import re
                # Remove assistant header if present
                generated_text = re.sub(r'^assistant\s*\n*\s*', '', generated_text, flags=re.IGNORECASE)
                generated_text = generated_text.strip()
                
                # Also try to extract by removing the input text from decoded output
                # Sometimes tokenizer decoding includes the input, so we strip it
                if input_text in decoded_full:
                    # Remove the input part
                    generated_text_alt = decoded_full.replace(input_text, "", 1).strip()
                    # Remove assistant header if present
                    generated_text_alt = re.sub(r'^assistant\s*\n*\s*', '', generated_text_alt, flags=re.IGNORECASE)
                    generated_text_alt = generated_text_alt.strip()
                    # Use the longer one (more complete generation)
                    if len(generated_text_alt) > len(generated_text):
                        generated_text = generated_text_alt
                
                all_outputs.append(generated_text)  # Store only generated part
                
                # Evaluate correctness using only the generated part
                is_correct = self._check_correctness(example, generated_text, subtask_name, is_decoded=True)
                if is_correct:
                    correct += 1
                total += 1
                
                # Debug output for first example
                if debug and total == 1:
                    expected_path = example.get('path', [])
                    expected_path_str = ">".join([str(p) for p in expected_path])
                    print(f"\n    DEBUG - First example evaluation:")
                    print(f"      Input: {input_text}")
                    print(f"      Expected path: {expected_path_str}")
                    print(f"      Full model output: {decoded_full}")
                    print(f"      Generated part only: '{generated_text}'")
                    print(f"      Generated length: {len(generated_text)} chars")
                    print(f"      Full path in generated? {expected_path_str in generated_text}")
                    if len(expected_path) >= 7:
                        path_7 = ">".join([str(p) for p in expected_path[:7]])
                        print(f"      First 7 nodes ({path_7}) in generated? {path_7 in generated_text}")
                    if len(expected_path) >= 5:
                        path_5 = ">".join([str(p) for p in expected_path[:5]])
                        print(f"      First 5 nodes ({path_5}) in generated? {path_5 in generated_text}")
                    # Check what sequences FROM THE START are found
                    found_sequences_from_start = []
                    for seq_len in range(5, min(10, len(expected_path) + 1)):
                        seq = expected_path[0:seq_len]  # Always from start
                        seq_str = ">".join([str(p) for p in seq])
                        if seq_str in generated_text:
                            found_sequences_from_start.append(seq_str)
                    if found_sequences_from_start:
                        print(f"      Found sequences FROM START: {found_sequences_from_start}")
                    else:
                        print(f"      Found sequences FROM START: NONE")
                        # Also check what random sequences might be found (for debugging)
                        random_sequences = []
                        for seq_len in range(3, 6):
                            for start_idx in range(1, len(expected_path) - seq_len + 1):  # Not from start
                                seq = expected_path[start_idx:start_idx + seq_len]
                                seq_str = ">".join([str(p) for p in seq])
                                if seq_str in generated_text:
                                    random_sequences.append(seq_str)
                        if random_sequences:
                            print(f"      Found random subsequences (NOT from start): {random_sequences[:3]}")
                    print(f"      Correct: {is_correct}")
                    if ablated_heads:
                        print(f"      Ablated heads: {ablated_heads}")
                    else:
                        print(f"      Mode: BASELINE (no heads ablated)")
                    
            except Exception as e:
                # Skip examples that fail
                if debug:
                    print(f"    ERROR processing example: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        
        if debug:
            print(f"\n    Evaluation summary:")
            print(f"      Correct: {correct}/{total}")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Sample outputs: {all_outputs[:2]}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _evaluate_subtask_bleu(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        ablated_heads: Optional[List[Tuple[int, int]]] = None,
        debug: bool = False
    ) -> Dict[str, float]:
        """Evaluate using BLEU score for free-form text generation."""
        # BLEU_AVAILABLE should always be True now (we have fallback)
        
        bleu_scores = []
        all_outputs = []
        all_references = []
        
        for example in examples:
            try:
                # Convert example to input format
                input_text = self._format_example(example)
                
                # Debug: show formatted prompt for first example
                if debug and len(bleu_scores) == 0:
                    print(f"\n    DEBUG - Formatted prompt (first 500 chars):")
                    print(f"      {input_text[:500]}...")
                    if len(input_text) > 500:
                        print(f"      ... (total length: {len(input_text)} chars)")
                
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
                
                # Generate with or without ablation
                with torch.no_grad():
                    if ablated_heads:
                        output = self._generate_with_ablation(input_ids, ablated_heads)
                    else:
                        output = self.model.generate(
                            input_ids,
                            max_new_tokens=150,  # Longer for free-form answers
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
                
                all_outputs.append(generated_text)
                
                # Get reference answer and extract simplified version
                reference_full = example.get("answer", "") or example.get("subquestion_answer", "")
                if not reference_full:
                    continue
                
                # Extract simplified reference: "yes", "no", or "unanswerable"
                reference_simplified = self._extract_simplified_answer(reference_full)
                all_references.append(reference_simplified)
                
                # Also extract simplified answer from generated text
                generated_simplified = self._extract_simplified_answer(generated_text)
                
                # Always calculate BLEU on full reference text for more nuanced scoring
                reference_tokens = reference_full.lower().split()
                generated_tokens_list = generated_text.lower().split()
                
                if USE_NLTK_BLEU:
                    smoothing = SmoothingFunction().method1
                    bleu_full = sentence_bleu(
                        [reference_tokens],
                        generated_tokens_list,
                        smoothing_function=smoothing
                    )
                elif USE_EVALUATE_LIB:
                    result = bleu_metric.compute(
                        predictions=[generated_tokens_list],
                        references=[[reference_tokens]]
                    )
                    bleu_full = result.get("bleu", 0.0)
                else:
                    bleu_full = simple_bleu(reference_tokens, generated_tokens_list)
                
                # Check if reference itself is already simplified (short answer like "yes", "no")
                # Only give exact match bonus if reference is already simple
                reference_is_simple = (
                    len(reference_full.strip().split()) <= 3 and 
                    reference_full.strip().lower() in ["yes", "no", "unanswerable", "it is not possible to tell"]
                )
                
                # If simplified answers match AND reference is already simple, use exact match
                if reference_simplified in ["yes", "no", "unanswerable"]:
                    if generated_simplified.lower() == reference_simplified.lower():
                        if reference_is_simple:
                            # Reference is already simple, exact match = 100%
                            bleu = 1.0
                        else:
                            # Reference is longer, use BLEU but with bonus for exact match on simplified
                            # Use max of BLEU and 0.9 (exact match on simplified gets high score)
                            bleu = max(bleu_full, 0.9)
                    else:
                        # No exact match, use BLEU on full reference
                        bleu = bleu_full
                else:
                    # Reference not simplified, use BLEU on full reference
                    bleu = bleu_full
                
                bleu_scores.append(bleu)
                
                # Debug output for first example
                if debug and len(bleu_scores) == 1:
                    print(f"\n    DEBUG - First example evaluation:")
                    print(f"      Input: {input_text[:200]}...")
                    print(f"      Reference (full): {reference_full}")
                    print(f"      Reference (simplified): {reference_simplified}")
                    print(f"      Generated (full): {generated_text[:200]}...")
                    print(f"      Generated (simplified): {generated_simplified}")
                    print(f"      Exact match (Yes/No): {generated_simplified.lower() == reference_simplified.lower() if reference_simplified in ['yes', 'no', 'unanswerable'] else 'N/A'}")
                    print(f"      BLEU score: {bleu:.4f}")
                    if ablated_heads:
                        print(f"      Ablated heads: {ablated_heads}")
                    else:
                        print(f"      Mode: BASELINE (no heads ablated)")
                
            except Exception as e:
                if debug:
                    print(f"    ERROR processing example: {e}")
                continue
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        if debug:
            print(f"\n    Evaluation summary:")
            print(f"      Average BLEU: {avg_bleu:.4f}")
            print(f"      BLEU scores: {bleu_scores[:5]}")
            print(f"      Sample outputs: {all_outputs[:2]}")
        
        return {
            "accuracy": avg_bleu,  # Use BLEU as "accuracy" metric
            "bleu_score": avg_bleu,
            "correct": len(bleu_scores),
            "total": len(examples)
        }
    
    def _evaluate_atomic_task(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        ablated_heads: Optional[List[Tuple[int, int]]] = None,
        debug: bool = False
    ) -> Dict[str, float]:
        """Evaluate atomic task using exact match on multiple-choice answers."""
        correct = 0
        total = 0
        all_outputs = []
        all_extracted = []
        
        for example in examples:
            try:
                # Format example using atomic task prompt
                # Use subtask_name (correct task type) instead of example's task_type field
                input_text = self._format_atomic_task_example(example, task_type=subtask_name)
                
                # Debug: show formatted prompt for first example
                if debug and total == 0:
                    print(f"\n    DEBUG - Formatted prompt (first 500 chars):")
                    print(f"      {input_text[:500]}...")
                    if len(input_text) > 500:
                        print(f"      ... (total length: {len(input_text)} chars)")
                
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
                
                # Generate with or without ablation
                with torch.no_grad():
                    if ablated_heads:
                        output = self._generate_with_ablation(input_ids, ablated_heads)
                    else:
                        output = self.model.generate(
                            input_ids,
                            max_new_tokens=100,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                
                # Decode and extract generated text
                decoded_full = self.tokenizer.decode(output[0], skip_special_tokens=True)
                input_length = len(input_ids[0])
                generated_tokens = output[0][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Clean up
                generated_text = re.sub(r'^assistant\s*\n*\s*', '', generated_text, flags=re.IGNORECASE)
                generated_text = generated_text.strip()
                
                if input_text in decoded_full:
                    generated_text_alt = decoded_full.replace(input_text, "", 1).strip()
                    generated_text_alt = re.sub(r'^assistant\s*\n*\s*', '', generated_text_alt, flags=re.IGNORECASE)
                    generated_text_alt = generated_text_alt.strip()
                    if len(generated_text_alt) > len(generated_text):
                        generated_text = generated_text_alt
                
                all_outputs.append(generated_text)
                
                # Get correct answer
                correct_answer = example.get("answer", "")
                
                # Extract answer from model output
                extracted_answer = self._extract_atomic_answer(generated_text, example)
                all_extracted.append(extracted_answer)
                
                # Exact match comparison (case-insensitive, strip whitespace)
                is_correct = self._compare_atomic_answers(extracted_answer, correct_answer)
                if is_correct:
                    correct += 1
                total += 1
                
                # Debug output for first example
                if debug and total == 1:
                    print(f"\n    DEBUG - First example evaluation:")
                    print(f"      Question: {example.get('question', '')[:150]}...")
                    print(f"      Correct answer: '{correct_answer}'")
                    print(f"      Generated text: '{generated_text[:200]}...'")
                    print(f"      Extracted answer: '{extracted_answer}'")
                    print(f"      Is correct: {is_correct}")
                    if ablated_heads:
                        print(f"      Ablated heads: {ablated_heads}")
                    else:
                        print(f"      Mode: BASELINE (no heads ablated)")
                
            except Exception as e:
                if debug:
                    import traceback
                    print(f"    ERROR processing example: {e}")
                    traceback.print_exc()
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        
        if debug:
            print(f"\n    Evaluation summary:")
            print(f"      Correct: {correct}/{total}")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Sample outputs: {all_outputs[:2]}")
            print(f"      Sample extracted: {all_extracted[:2]}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _format_atomic_task_example(self, example: Dict[str, Any], task_type: Optional[str] = None) -> str:
        """Format an atomic task example using task-specific prompt template."""
        question = example.get("question", "")
        
        # Use provided task_type if available, otherwise try from example
        if task_type is None:
            task_type = example.get("task_type", "")
        
        # Validate task_type - skip if it looks like options string (e.g., '">", "<", "=", "unknown"')
        # Options strings are typically short and contain only symbols/quotes
        if task_type and task_type not in ATOMIC_TASK_PROMPTS:
            # Check if it looks like an options list (short, contains quotes/symbols, no "reasoning" keyword)
            if (len(task_type) < 30 and 
                ('"' in task_type or "'" in task_type or task_type.startswith('[')) and
                'reasoning' not in task_type.lower()):
                task_type = ""  # Reset to empty, will detect from question
        
        # Safety check: if task_type is missing or invalid, try to detect from question
        if not task_type or task_type not in ATOMIC_TASK_PROMPTS:
            # Try to detect task type from question content
            q_lower = question.lower()
            if any(x in q_lower for x in ['taller', 'shorter', 'older', 'younger', 'heavier', 'fastest', 'tallest']):
                task_type = "Transitive reasoning-scalar max"
            elif 'options:' in q_lower and ("'>'," in question or "'<'," in question or '">", "<"' in question):
                task_type = "Transitive reasoning-symbolic inequality"
            elif any(x in q_lower for x in ['before', 'after', 'first', 'last', 'happens']):
                task_type = "Transitive reasoning-temporal order"
            elif any(x in q_lower for x in ['north', 'south', 'left', 'right', 'inside', 'above', 'below']):
                task_type = "Transitive reasoning-spatial containment"
            elif any(x in q_lower for x in ['all ', 'every ', 'if someone', 'no ', 'must be true']):
                task_type = "Transitive reasoning-subset/implication"
            elif any(x in q_lower for x in ['manages', 'reports to', 'supervisor', 'parent', 'ancestor', 'hierarchy']):
                task_type = "Transitive reasoning-hierarchy"
        
        # Get task-specific prompt or fall back to generic
        if task_type in ATOMIC_TASK_PROMPTS:
            prompt = ATOMIC_TASK_PROMPTS[task_type].format(question=question)
        else:
            # Try to find by partial match (only if task_type is non-empty)
            prompt = None
            if task_type:
                for key in ATOMIC_TASK_PROMPTS:
                    if key.lower() in task_type.lower() or task_type.lower() in key.lower():
                        prompt = ATOMIC_TASK_PROMPTS[key].format(question=question)
                        break
            if prompt is None:
                prompt = ATOMIC_TASK_PROMPT_GENERIC.format(question=question)
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            return formatted
        else:
            return prompt
    
    def _extract_atomic_answer(self, generated_text: str, example: Dict[str, Any]) -> str:
        """Extract the answer from model output for atomic tasks."""
        text = generated_text.strip()
        
        # Try to extract from [ "answer": "..." ] format
        match = re.search(r'\[\s*"answer"\s*:\s*"([^"]+)"\s*\]', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try alternate format: [ "answer": '...' ]
        match = re.search(r"\[\s*['\"]answer['\"]\s*:\s*['\"]([^'\"]+)['\"]\s*\]", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try JSON-like format: { "answer": "..." }
        match = re.search(r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try to find answer after "answer:" or "Answer:"
        match = re.search(r'answer\s*:\s*"?([^\n\[\]"]+)"?', text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up quotes
            answer = answer.strip('"\'')
            return answer
        
        # For symbolic inequality - look for standalone symbols
        task_type = example.get("task_type", "")
        if "symbolic" in task_type.lower() or "inequality" in task_type.lower():
            # Check for symbols at start of text or after common patterns
            for symbol in ['>', '<', '=', 'unknown']:
                # Check various patterns
                if text.strip() == symbol:
                    return symbol
                if text.startswith(f'"{symbol}"') or text.startswith(f"'{symbol}'"):
                    return symbol
                if f'"{symbol}"' in text or f"'{symbol}'" in text:
                    return symbol
                # Check for symbol followed by punctuation or whitespace
                if re.search(rf'(?:^|[\s\[\("])({re.escape(symbol)})(?:[\s\]\)"\',.]|$)', text):
                    return symbol
        
        # Get the options from the question to match against
        question = example.get("question", "")
        options_match = re.search(r'Options:\s*\[([^\]]+)\]', question)
        if options_match:
            options_text = options_match.group(1)
            # Parse options - handle various quote formats
            # Pattern: 'option', "option", or just option
            options = []
            # Try to find quoted options first (both single and double quotes)
            quoted_opts = re.findall(r"['\"]([^'\"]+)['\"]", options_text)
            if quoted_opts:
                options = quoted_opts
            else:
                # Fallback: split by comma and strip
                options = [opt.strip().strip('"\'') for opt in options_text.split(',')]
            
            # Clean up options
            options = [opt.strip() for opt in options if opt.strip()]
            
            # Check if any option appears in the generated text
            text_lower = text.lower()
            for opt in options:
                opt_clean = opt.strip()
                opt_lower = opt_clean.lower()
                # Check for exact match
                if opt_lower == text_lower.strip():
                    return opt_clean
                # Check for quoted option
                if f'"{opt_clean}"' in text or f"'{opt_clean}'" in text:
                    return opt_clean
                # For short options (like >, <, =), check if they appear standalone
                if len(opt_clean) <= 10:
                    # For symbolic options, be more careful
                    if opt_clean in ['>', '<', '=', 'unknown']:
                        # Already handled above in the symbolic section
                        pass
                    elif opt_lower in text_lower:
                        return opt_clean
        
        # Fallback: return first line or first few words
        first_line = text.split('\n')[0].strip()
        if len(first_line) > 0 and len(first_line) < 100:
            return first_line.strip('"\'[]')
        
        return text[:50].strip('"\'[]') if text else ""
    
    def _compare_atomic_answers(self, extracted: str, correct: str) -> bool:
        """Compare extracted answer with correct answer (flexible matching)."""
        if not extracted or not correct:
            return False
        
        # Normalize both answers
        extracted_norm = extracted.lower().strip().strip('"\'')
        correct_norm = correct.lower().strip().strip('"\'')
        
        # Exact match
        if extracted_norm == correct_norm:
            return True
        
        # Check if correct answer is contained in extracted (for partial matches)
        if correct_norm in extracted_norm:
            return True
        
        # For symbolic answers like ">", "<", "=", "unknown"
        if correct_norm in ['>', '<', '=', 'unknown']:
            # Check if the symbol appears in the extracted text
            if correct_norm in extracted_norm:
                return True
            # Check for word equivalents
            if correct_norm == '>' and any(w in extracted_norm for w in ['greater', 'more than', 'larger']):
                return True
            if correct_norm == '<' and any(w in extracted_norm for w in ['less', 'smaller', 'fewer']):
                return True
            if correct_norm == '=' and any(w in extracted_norm for w in ['equal', 'same']):
                return True
        
        # For Yes/No answers
        if correct_norm in ['yes', 'no']:
            if correct_norm == 'yes' and any(w in extracted_norm for w in ['yes', 'true', 'correct', 'necessarily true']):
                return True
            if correct_norm == 'no' and any(w in extracted_norm for w in ['no', 'false', 'incorrect', 'not necessarily']):
                return True
        
        return False
        
    def _generate_with_ablation(
        self,
        input_ids: torch.Tensor,
        ablated_heads: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Generate with specific heads ablated using hooks."""
        
        # Store original attention weights
        hooks = []
        
        def create_hook(layer_idx, head_idx):
            def hook_fn(module, input, output):
                # Assuming output is (batch, heads, seq, seq) for attention weights
                # Or (batch, seq, hidden) for attention output
                if isinstance(output, tuple):
                    # Some models return (attn_output, attn_weights)
                    attn_output = output[0]
                else:
                    attn_output = output
                    
                # Get dimensions
                if len(attn_output.shape) == 3:  # (batch, seq, hidden)
                    batch_size, seq_len, hidden_dim = attn_output.shape
                    n_heads = self.model.config.num_attention_heads
                    head_dim = hidden_dim // n_heads
                    
                    # Reshape to separate heads
                    attn_output = attn_output.view(batch_size, seq_len, n_heads, head_dim)
                    # Zero out the specified head
                    attn_output[:, :, head_idx, :] = 0
                    # Reshape back
                    attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
                    
                    if isinstance(output, tuple):
                        return (attn_output,) + output[1:]
                    return attn_output
                return output
            return hook_fn
        
        # Register hooks for each head to ablate
        try:
            for layer_idx, head_idx in ablated_heads:
                # Find the attention module for this layer
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # Llama/Mistral style
                    attn_module = self.model.model.layers[layer_idx].self_attn
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    # GPT-2 style
                    attn_module = self.model.transformer.h[layer_idx].attn
                else:
                    # Try to find it generically
                    continue
                    
                hook = attn_module.register_forward_hook(create_hook(layer_idx, head_idx))
                hooks.append(hook)
            
            # Generate with hooks active
            output = self.model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return output
            
       
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """
        Format example for model input.
        
        Supports backward-chaining, CognitiveMirrors, and atomic task formats.
        """
        # Check if this is an atomic task example (has task_type field)
        # Validate task_type is actually a valid atomic task type (not corrupted data)
        example_task_type = example.get("task_type", "")
        if example_task_type and example_task_type in ATOMIC_TASK_TYPES.values():
            return self._format_atomic_task_example(example, task_type=example_task_type)
        
        # Check if this is a CognitiveMirrors example
        if "subquestion" in example or ("question" in example and "subquestion_answer" in example):
            return self._format_cognitive_mirrors_example(example)
        
        # Backward-chaining format
        if "edges" in example:
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"]])
            goal = example.get("goal", "?")
            # Format: edges|goal:
            raw_input = f"{edges_str}|{goal}:"
            
            # Check if tokenizer has a chat template (for instruct models)
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                # Use chat template for instruct models
                # Find root node (node that appears as source but never as target)
                source_nodes = set([e[0] for e in example.get("edges", [])])
                target_nodes = set([e[1] for e in example.get("edges", [])])
                root_nodes = source_nodes - target_nodes
                root_node = list(root_nodes)[0] if root_nodes else "?"
                
                # Format edges as list: "A1>B1, A2>B2, ..., An>Bn"
                # This matches the training format from the paper
                # The model receives: edge list, goal, root, and should predict the path
                
                user_content = f"Edge list: {edges_str}\n"
                user_content += f"Goal: {goal}\n"
                user_content += f"Root: {root_node}\n"
                user_content += f"Path:"
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are solving a backward-chaining path-finding problem in a tree. "
                            "Given an edge list of a directed tree, a goal node, and the root node, "
                            "you must find the unique path from the root to the goal. "
                            "The path is a sequence of node numbers separated by '>', starting from the root and ending at the goal. "
                            "Output ONLY the path sequence (e.g., '10>7>3>5>0>1>2>6>14>12>15>9>11>13'), nothing else."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
                # Apply chat template and return as text (not tokenized)
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                return formatted
            else:
                # For base models, use raw format
                return raw_input
        return str(example)
    
    def _extract_simplified_answer(self, answer_text: str) -> str:
        """
        Extract simplified answer from full answer text.
        Returns: "yes", "no", or "unanswerable"
        """
        answer_lower = answer_text.lower().strip()
        
        # Check for "no" patterns
        no_patterns = [
            r'\bno\b',
            r'\bnot\b.*factual',
            r'\bnot\b.*true',
            r'\bnot\b.*correct',
            r'is\s+not',
            r'are\s+not',
            r'does\s+not',
            r'do\s+not',
        ]
        for pattern in no_patterns:
            if re.search(pattern, answer_lower):
                return "no"
        
        # Check for "yes" patterns
        yes_patterns = [
            r'\byes\b',
            r'\bis\b.*factual',
            r'\bis\b.*true',
            r'\bis\b.*correct',
            r'is\s+factual',
            r'is\s+true',
            r'is\s+correct',
        ]
        for pattern in yes_patterns:
            if re.search(pattern, answer_lower):
                return "yes"
        
        # Check for unanswerable patterns
        unanswerable_patterns = [
            r'\bunanswerable\b',
            r'\bcannot\s+determine\b',
            r'\bcannot\s+tell\b',
            r'\bnot\s+possible\s+to\s+tell\b',
            r'\bnot\s+enough\s+information\b',
            r'\binsufficient\s+information\b',
        ]
        for pattern in unanswerable_patterns:
            if re.search(pattern, answer_lower):
                return "unanswerable"
        
        # Default: try to infer from first few words
        first_words = answer_lower.split()[:5]
        if any(word in ['no', 'not'] for word in first_words):
            return "no"
        elif any(word in ['yes'] for word in first_words):
            return "yes"
        
        # If we can't determine, return original (will use full text for BLEU)
        return answer_text
    
    def _format_cognitive_mirrors_example(self, example: Dict[str, Any]) -> str:
        """Format CognitiveMirrors example for model input using the specified prompt format."""
        question = example.get("question", "")
        subquestion = example.get("subquestion", "")
        
        # Build prior knowledge (CoT) from previous subquestions if available
        prior_knowledge = ""
        if "full_example" in example and "generated" in example["full_example"]:
            # Get previous subquestions for context (CoT format)
            generated = example["full_example"]["generated"]
            for item in generated:
                if item.get("subquestion") != subquestion:  # Exclude current subquestion
                    prior_knowledge += f"Q: {item.get('subquestion', '')}\nA: {item.get('answer', '')}\n"
        
        if not prior_knowledge.strip():
            prior_knowledge = "No prior knowledge available."
        
        # Construct the prompt according to the specified format
        prompt = (
            "Prompt: You are an expert in analytical and logical reasoning. "
            "You will be given a main question and prior knowledge in chain-of-thought (CoT) format. "
            "Your task is to answer a follow-up subquestion using the information provided.\n"
            "Here is the main question:.\n"
            "<main question> " + question + " </main question>\n"
            "Here is the prior knowledge in chain-of-thought (CoT) format.\n"
            "<prior knowledge> " + prior_knowledge.strip() + " </prior knowledge>\n"
            "Here is the subquestion:\n"
            "<subquestion> " + subquestion + " </subquestion>\n"
            "Instructions:\n"
            "Answer the subquestion carefully.\n"
            "You can use the information in the prior knowledge to help you answer the subquestion.\n"
            "Your response should be clear and concise.\n"
            "Stick to factual reasoning based on provided CoT.\n"
            "Do not include any explanation, commentary, or code.\n"
            "Do not output anything after the closing square bracket ']'.\n"
            "Only output your final answer using this format: [ \"answer\": \"<Your answer here>\" ]\n"
            "Your answer:"
        )
        
        # Check if tokenizer has a chat template (for instruct models)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            # Use chat template but with the new prompt format
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            formatted = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            return formatted
        else:
            # For base models, return the prompt directly
            return prompt
    
    def _check_correctness(
        self,
        example: Dict[str, Any],
        output: Any,  # Can be Tensor or decoded string
        subtask_name: str,
        is_decoded: bool = False
    ) -> bool:
        # Decode output if it's a tensor
        if not is_decoded:
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            decoded = output  # Already decoded string
        
        # Check based on subtask type
        if subtask_name in ["path_finding", "node_traversal", "backward_chain_step"]:
            # For path-finding tasks, check if the path appears in output
            expected_path = example.get("path", [])
            if len(expected_path) == 0:
                return False
            
            # Convert expected path to string format
            expected_path_str = ">".join([str(p) for p in expected_path])
            
            # ULTRA STRICT: Check if the full path sequence appears with > separators
            # This is the only reliable way to check correctness
            if expected_path_str in decoded:
                return True
            
            # Check if at least first 7 nodes in sequence appear (more than half)
            # This ensures we're getting a substantial portion of the path
            if len(expected_path) >= 7:
                path_start = ">".join([str(p) for p in expected_path[:7]])
                if path_start in decoded:
                    return True
            
            # Check if at least first 5 nodes in sequence appear
            if len(expected_path) >= 5:
                path_start = ">".join([str(p) for p in expected_path[:5]])
                if path_start in decoded:
                    return True
            
            # ULTRA STRICT FALLBACK: Require path sequence FROM THE BEGINNING
            # We ONLY check sequences starting from index 0 (the root of the path)
            # This prevents matching on random edge patterns from the input
            
            # Check sequences starting from the beginning of the path only
            # We check from longest to shortest to find the best match
            for seq_len in range(min(10, len(expected_path)), 4, -1):  # Check 10 down to 5
                # Always start from index 0 (beginning of path - the root)
                seq = expected_path[0:seq_len]
                seq_str = ">".join([str(p) for p in seq])
                # Check if this sequence appears in decoded
                if seq_str in decoded:
                    # Found a valid sequence from the start
                    return True
            
            # If we get here, no valid sequence from the start was found
            # The model did not generate the path correctly
            return False
            
        elif subtask_name in ["goal_identification", "edge_parsing"]:
            # For goal/edge tasks, check if goal appears
            goal = example.get("goal")
            if goal is not None:
                # Check if goal number appears (not just as part of another number)
                goal_str = str(goal)
                # Look for goal as standalone or with > separator
                if f">{goal_str}" in decoded or f"{goal_str}:" in decoded or decoded.strip().endswith(goal_str):
                    return True
            return False
            
        elif subtask_name in ["graph_construction", "token_prediction"]:
            # For construction/prediction, check if output contains graph-related tokens
            # Look for edge-like patterns (number>number)
            import re
            edge_pattern = r'\d+>\d+'
            if re.search(edge_pattern, decoded):
                return True
            return False
        
        # Default: check if output is non-empty and contains numbers
        # (to avoid passing empty or irrelevant outputs)
        if len(decoded.strip()) == 0:
            return False
        # Check if it contains at least one number (indicating some graph-related content)
        import re
        return bool(re.search(r'\d+', decoded))


class CausalPatchingScorer(HeadScorer):
    """
    Score heads using causal attention patching.
    
    Replace head activations with baseline and measure effect on output.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        super().__init__(model, tokenizer, device)
        self.cache = {}
    
    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str
    ) -> HeadScore:
        # Get clean (correct) examples
        clean_examples = [ex for ex in examples if self._is_clean(ex)]
        corrupted_examples = [ex for ex in examples if not self._is_clean(ex)]
        
        if len(clean_examples) == 0 or len(corrupted_examples) == 0:
            # Use all examples as both clean and corrupted
            clean_examples = examples
            corrupted_examples = examples
        
        baseline_diff = self._get_logit_difference(clean_examples, corrupted_examples, subtask_name)
        
        # Get patched logit difference
        patched_diff = self._get_patched_logit_difference(
            clean_examples, corrupted_examples, layer, head, subtask_name
        )
        
        # Score is the normalized effect of patching
        if baseline_diff != 0:
            score = abs(patched_diff - baseline_diff) / abs(baseline_diff)
        else:
            score = abs(patched_diff)
        
        confidence = min(len(examples) / 10.0, 1.0)
        
        return HeadScore(
            layer=layer,
            head=head,
            score=score,
            confidence=confidence,
            method="causal_patching",
            metadata={
                "baseline_logit_diff": baseline_diff,
                "patched_logit_diff": patched_diff,
                "n_examples": len(examples)
            }
        )
    
    def _is_clean(self, example: Dict[str, Any]) -> bool:
        # Simplified - should check actual correctness
        return True
    
    def _get_logit_difference(
        self,
        clean_examples: List[Dict[str, Any]],
        corrupted_examples: List[Dict[str, Any]],
        subtask_name: str
    ) -> float:
        # Simplified implementation
        return 1.0
    
    def _get_patched_logit_difference(
        self,
        clean_examples: List[Dict[str, Any]],
        corrupted_examples: List[Dict[str, Any]],
        layer: int,
        head: int,
        subtask_name: str
    ) -> float:
        # Simplified implementation
        return 0.5


class MutualInfoScorer(HeadScorer):
    """
    Score heads using mutual information between head activations and subtask labels.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        super().__init__(model, tokenizer, device)
    
    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str
    ) -> HeadScore:
        # Collect activations and labels
        activations = []
        labels = []
        
        for example in examples:
            # Get head activations
            act = self._get_head_activation(layer, head, example)
            if act is not None:
                activations.append(act)
                # Get label for this subtask
                label = self._get_subtask_label(example, subtask_name)
                labels.append(label)
        
        if len(activations) < 2:
            return HeadScore(
                layer=layer, head=head, score=0.0, confidence=0.0,
                method="mutual_info", metadata={"error": "insufficient_data"}
            )
        
        # Calculate mutual information
        activations = np.array(activations)
        labels = np.array(labels)
        
        # Discretize activations for MI calculation
        act_binned = self._discretize(activations)
        label_binned = self._discretize(labels) if labels.dtype == float else labels
        
        # Calculate MI
        mi_score = self._mutual_information(act_binned, label_binned)
        
        confidence = min(len(examples) / 20.0, 1.0)
        
        return HeadScore(
            layer=layer,
            head=head,
            score=mi_score,
            confidence=confidence,
            method="mutual_info",
            metadata={
                "n_examples": len(examples),
                "mean_activation": float(np.mean(activations)),
                "std_activation": float(np.std(activations))
            }
        )
    
    def _get_head_activation(
        self,
        layer: int,
        head: int,
        example: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        # This would need to hook into model forward pass
        # Simplified version
        return np.random.randn(10)  # Placeholder
    
    def _get_subtask_label(self, example: Dict[str, Any], subtask_name: str) -> Any:
        if subtask_name == "path_finding":
            return len(example.get("path", []))
        elif subtask_name == "goal_identification":
            return example.get("goal", 0)
        return 0
    
    def _discretize(self, values: np.ndarray, n_bins: int = 10) -> np.ndarray:
        if values.dtype == float:
            _, bins = np.histogram(values, bins=n_bins)
            return np.digitize(values, bins) - 1
        return values
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        # Use scipy's mutual information
        try:
            from sklearn.metrics import mutual_info_score
            return mutual_info_score(x, y)
        except ImportError:
            # Fallback: simple correlation
            if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                return abs(np.corrcoef(x, y)[0, 1])
            return 0.0


def create_scorer(
    method: str,
    model,
    tokenizer,
    device: str = "cuda",
    **kwargs
) -> HeadScorer:
    if method == "ablation":
        return AblationScorer(model, tokenizer, device, **kwargs)
    elif method == "causal_patching":
        return CausalPatchingScorer(model, tokenizer, device)
    elif method == "mutual_info":
        return MutualInfoScorer(model, tokenizer, device)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

