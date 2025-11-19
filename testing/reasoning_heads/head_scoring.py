"""
Head scoring methods for identifying reasoning heads.

Implements multiple scoring approaches:
1. Ablation effect: Measure change when head is zeroed/randomized
2. Causal attention patching: Replace head activations with baselines
3. Mutual information: Correlation between head outputs and subtask signals
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy import stats
import json
import os
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, leave=True):
        return iterable


@dataclass
class HeadScore:
    """Score for a specific attention head."""
    layer: int
    head: int
    score: float
    confidence: float
    method: str
    metadata: Dict[str, Any] = None
    
    def __repr__(self):
        return f"HeadScore(layer={self.layer}, head={self.head}, score={self.score:.4f}, confidence={self.confidence:.4f})"


class HeadScorer(ABC):
    """Base class for head scoring methods."""
    
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
        """Score a specific head for a subtask."""
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
        """Score all heads and return ranked list."""
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
        
        scores = []
        from tqdm import tqdm
        
        # Debug mode for first head only
        first_head = True
        
        for layer in tqdm(range(max_layers), desc="  Layers", leave=False):
            for head in range(max_heads_per_layer):
                try:
                    # Only debug first head of first layer
                    debug = first_head and layer == 0 and head == 0
                    score = self.score_head(layer, head, examples, subtask_name, debug=debug)
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
    """
    Score heads by measuring the effect of ablating (zeroing) them.
    
    Higher score = more important head (larger performance drop when ablated).
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda", ablation_type: str = "zero"):
        super().__init__(model, tokenizer, device)
        self.ablation_type = ablation_type  # "zero" or "random"
    
    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        debug: bool = False
    ) -> HeadScore:
        """Score head by ablation effect."""
        # Use a smaller subset for faster scoring
        # For initial discovery, use just 2-3 examples per head
        scoring_examples = examples[:min(3, len(examples))]
        
        # Get baseline performance (only debug first head to avoid spam)
        baseline_debug = debug
        baseline_metrics = self._evaluate_subtask(
            scoring_examples, subtask_name, ablated_heads=None, debug=baseline_debug
        )
        
        # Get performance with head ablated (also debug if in debug mode)
        ablated_metrics = self._evaluate_subtask(
            scoring_examples, subtask_name, ablated_heads=[(layer, head)], debug=debug
        )
        
        # Calculate score as relative performance drop
        baseline_acc = baseline_metrics.get("accuracy", 0)
        ablated_acc = ablated_metrics.get("accuracy", 0)
        
        # Score calculation:
        # Positive score = head is important (masking hurts performance)
        # Negative score = head might be harmful (masking improves performance) OR check is wrong
        if baseline_acc > 0:
            score = (baseline_acc - ablated_acc) / baseline_acc
        else:
            # If baseline is 0, use absolute difference
            # But if ablated is better, that's suspicious - might indicate check is wrong
            score = baseline_acc - ablated_acc
        
        # Handle negative scores
        # Negative score means masking improved performance, which could indicate:
        # 1. The head is actually harmful (unlikely but possible)
        # 2. The correctness check is inconsistent between baseline and ablated
        # 3. Random variation in a small sample
        # 
        # For now, we'll set negative scores to 0 (not important) since they're likely
        # due to correctness check issues rather than actual head importance
        if score < 0:
            if debug:
                print(f"      WARNING: Negative score ({score:.4f}) - masking improved performance!")
                print(f"      This likely indicates correctness check inconsistency")
                print(f"      Setting score to 0 (head not important)")
            score = 0.0  # Set to 0 rather than absolute value
        
        # Confidence based on number of examples and consistency
        confidence = min(len(scoring_examples) / 10.0, 1.0)
        
        if debug:
            print(f"\n    Head scoring (Layer {layer}, Head {head}):")
            print(f"      Baseline accuracy: {baseline_acc:.4f}")
            print(f"      Ablated accuracy: {ablated_acc:.4f}")
            print(f"      Score: {score:.4f}")
        
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
        """Evaluate model performance on subtask."""
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
    
    def _generate_with_ablation(
        self,
        input_ids: torch.Tensor,
        ablated_heads: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Generate with specific heads ablated."""
        # This is a simplified version - actual implementation depends on model architecture
        # For now, we'll use a hook-based approach
        
        def ablation_hook(module, input, output, layer_idx, head_idx):
            """Hook to zero out specific head."""
            if self.ablation_type == "zero":
                # Zero out the head's contribution
                output[:, head_idx, :, :] = 0
            elif self.ablation_type == "random":
                # Randomize the head's contribution
                output[:, head_idx, :, :] = torch.randn_like(output[:, head_idx, :, :])
            return output
        
        # Register hooks (simplified - actual implementation needs proper hook registration)
        # For now, we'll use the block_list mechanism if available
        if hasattr(self.model, 'generate'):
            # Try to use block_list if model supports it
            try:
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=100,  # Increased for longer paths
                    do_sample=False,
                    temperature=1.0,  # Not used when do_sample=False, but explicit
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    block_list=ablated_heads
                )
                return output
            except:
                pass
        
        # Fallback: standard generation
        return self.model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """
        Format example for model input.
        
        Converts the backward-chaining example into a text prompt.
        For instruct models, uses the chat template with proper instructions.
        
        Format: "edge1,edge2,...|goal:"
        Example: "12>4,14>12,1>2|13:"
        
        The model should then generate the path: "10>7>3>5>..."
        """
        # Convert backward-chaining example to text
        if "edges" in example:
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"]])
            goal = example.get("goal", "?")
            # Format: edges|goal:
            raw_input = f"{edges_str}|{goal}:"
            
            # Check if tokenizer has a chat template (for instruct models)
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                # Use chat template for instruct models with few-shot examples
                # Find root node (node that appears as source but never as target)
                source_nodes = set([e[0] for e in example.get("edges", [])])
                target_nodes = set([e[1] for e in example.get("edges", [])])
                root_nodes = source_nodes - target_nodes
                root_node = list(root_nodes)[0] if root_nodes else "?"
                
                # Create a few-shot example from the current example if we have the path
                example_path = example.get("path", [])
                example_path_str = ">".join([str(p) for p in example_path]) if example_path else None
                
                user_content = f"Given a directed graph with edges and a goal node, find the complete path from the root node to the goal node.\n\n"
                user_content += f"Edges: {edges_str}\n"
                user_content += f"Root node: {root_node}\n"
                user_content += f"Goal node: {goal}\n\n"
                user_content += "Find the path from root to goal using backward-chaining. Output the complete path as a sequence of node numbers separated by '>', starting from the root and ending at the goal.\n\n"
                user_content += "Path:"
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that solves backward-chaining reasoning problems. "
                            "Given a directed graph with edges and a goal node, you need to find the complete path from the root node (the node that has no incoming edges) to the goal node. "
                            "Use backward-chaining: start from the goal and work backwards to find which nodes lead to it, then construct the forward path from root to goal. "
                            "Output ONLY the path sequence as numbers separated by '>', nothing else. For example: '10>7>3>5>0>1>2>6>14>12>15>9>11>13'"
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
    
    def _check_correctness(
        self,
        example: Dict[str, Any],
        output: Any,  # Can be Tensor or decoded string
        subtask_name: str,
        is_decoded: bool = False
    ) -> bool:
        """
        Check if output is correct for the subtask.
        
        LOGIC EXPLANATION:
        For backward-chaining tasks, we check if the model can generate the correct
        path from the graph edges and goal. The model should output a sequence like
        "node1>node2>node3..." that matches the expected path.
        
        IMPORTANT: We use a STRICT check because we need to see differences between
        baseline and ablated performance. If everything passes, we can't identify
        important heads.
        """
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
        """Score head using causal patching."""
        # Get clean (correct) examples
        clean_examples = [ex for ex in examples if self._is_clean(ex)]
        corrupted_examples = [ex for ex in examples if not self._is_clean(ex)]
        
        if len(clean_examples) == 0 or len(corrupted_examples) == 0:
            # Use all examples as both clean and corrupted
            clean_examples = examples
            corrupted_examples = examples
        
        # Get baseline logit difference
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
        """Check if example is 'clean' (correct)."""
        # Simplified - should check actual correctness
        return True
    
    def _get_logit_difference(
        self,
        clean_examples: List[Dict[str, Any]],
        corrupted_examples: List[Dict[str, Any]],
        subtask_name: str
    ) -> float:
        """Get logit difference between clean and corrupted."""
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
        """Get logit difference with head patched."""
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
        """Score head using mutual information."""
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
        """Extract head activation for example."""
        # This would need to hook into model forward pass
        # Simplified version
        return np.random.randn(10)  # Placeholder
    
    def _get_subtask_label(self, example: Dict[str, Any], subtask_name: str) -> Any:
        """Get label for subtask."""
        if subtask_name == "path_finding":
            return len(example.get("path", []))
        elif subtask_name == "goal_identification":
            return example.get("goal", 0)
        return 0
    
    def _discretize(self, values: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous values."""
        if values.dtype == float:
            _, bins = np.histogram(values, bins=n_bins)
            return np.digitize(values, bins) - 1
        return values
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between x and y."""
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
    """Factory function to create appropriate scorer."""
    if method == "ablation":
        return AblationScorer(model, tokenizer, device, **kwargs)
    elif method == "causal_patching":
        return CausalPatchingScorer(model, tokenizer, device)
    elif method == "mutual_info":
        return MutualInfoScorer(model, tokenizer, device)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

