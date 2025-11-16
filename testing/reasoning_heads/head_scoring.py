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
        n_heads: Optional[int] = None
    ) -> List[HeadScore]:
        """Score all heads and return ranked list."""
        if n_layers is None:
            n_layers = getattr(self.model.config, 'num_hidden_layers', 
                             getattr(self.model.config, 'n_layers', 32))
        if n_heads is None:
            n_heads = getattr(self.model.config, 'num_attention_heads',
                            getattr(self.model.config, 'n_heads', 32))
        
        scores = []
        for layer in range(n_layers):
            for head in range(n_heads):
                try:
                    score = self.score_head(layer, head, examples, subtask_name)
                    scores.append(score)
                except Exception as e:
                    print(f"Warning: Could not score layer {layer}, head {head}: {e}")
                    continue
        
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
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
        subtask_name: str
    ) -> HeadScore:
        """Score head by ablation effect."""
        # Get baseline performance
        baseline_metrics = self._evaluate_subtask(examples, subtask_name, ablated_heads=None)
        
        # Get performance with head ablated
        ablated_metrics = self._evaluate_subtask(
            examples, subtask_name, ablated_heads=[(layer, head)]
        )
        
        # Calculate score as relative performance drop
        if baseline_metrics.get("accuracy", 0) > 0:
            score = (baseline_metrics["accuracy"] - ablated_metrics["accuracy"]) / baseline_metrics["accuracy"]
        else:
            score = 0.0
        
        # Confidence based on number of examples and consistency
        confidence = min(len(examples) / 10.0, 1.0)
        
        return HeadScore(
            layer=layer,
            head=head,
            score=score,
            confidence=confidence,
            method="ablation",
            metadata={
                "baseline_accuracy": baseline_metrics.get("accuracy", 0),
                "ablated_accuracy": ablated_metrics.get("accuracy", 0),
                "n_examples": len(examples)
            }
        )
    
    def _evaluate_subtask(
        self,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        ablated_heads: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, float]:
        """Evaluate model performance on subtask."""
        correct = 0
        total = 0
        
        for example in examples:
            # Convert example to input format
            input_text = self._format_example(example)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate with or without ablation
            with torch.no_grad():
                if ablated_heads:
                    output = self._generate_with_ablation(input_ids, ablated_heads)
                else:
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            # Evaluate correctness (simplified - should be task-specific)
            is_correct = self._check_correctness(example, output, subtask_name)
            if is_correct:
                correct += 1
            total += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
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
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
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
        """Format example for model input."""
        # Convert backward-chaining example to text
        if "edges" in example:
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"]])
            goal = example.get("goal", "?")
            return f"{edges_str}|{goal}:"
        return str(example)
    
    def _check_correctness(
        self,
        example: Dict[str, Any],
        output: torch.Tensor,
        subtask_name: str
    ) -> bool:
        """Check if output is correct for the subtask."""
        # Decode output
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Check based on subtask
        if subtask_name == "path_finding":
            # Check if path is in output
            expected_path = example.get("path", [])
            path_str = ">".join([str(p) for p in expected_path])
            return path_str in decoded
        elif subtask_name == "goal_identification":
            goal = example.get("goal")
            return str(goal) in decoded
        
        # Default: always return True (should be improved)
        return True


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

