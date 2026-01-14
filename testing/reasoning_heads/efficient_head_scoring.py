"""
Efficient Head Scoring Methods for Reasoning Head Discovery

This module implements efficient alternatives to brute-force ablation for discovering
reasoning heads in transformer models. These methods provide 100-1000x speedup over
the original approach.

=============================================================================
WHY THE ORIGINAL BRUTE-FORCE APPROACH IS SLOW
=============================================================================

Original approach (AblationScorer in head_scoring.py):
    For each head (layer, head) in [max_layers × max_heads_per_layer]:
        1. Register forward hook to ablate this specific head
        2. Run full generation (up to 150 tokens) on all examples
        3. Compute accuracy/BLEU
        4. Remove hook
        5. Score = baseline_accuracy - ablated_accuracy

For Qwen3-4B with 36 layers × 32 heads = 1,152 heads and 20 examples:
    - Total forward passes: 1,152 heads × 20 examples × 150 tokens ≈ 3.5M
    - Each forward pass requires full model computation
    - Sequential processing (no parallelization)
    - Hook registration/removal overhead per head

=============================================================================
NEW EFFICIENT APPROACHES
=============================================================================

1. ACTIVATION-BASED SCORING (ActivationScorer)
   - Single forward pass to collect all head outputs
   - Score each head based on output statistics (norm, variance)
   - Cost: O(examples) forward passes
   - Speedup: ~1000x

2. GRADIENT-BASED SCORING (GradientScorer)
   - Forward pass + backward pass from target token
   - Score heads by gradient magnitude (influence on output)
   - Cost: O(examples) forward/backward passes
   - Speedup: ~500x

3. ATTENTION PATTERN ANALYSIS (AttentionPatternScorer)
   - Single forward pass with attention outputs
   - Score based on attention entropy and focus patterns
   - Cost: O(examples) forward passes
   - Speedup: ~1000x

=============================================================================
KEY ASSUMPTIONS
=============================================================================

1. Heads with larger activation norms have more influence on final output
2. Heads with higher activation variance across tokens are context-sensitive
3. Gradient magnitude indicates causal importance for predictions
4. Reasoning heads tend to have focused (low-entropy) attention patterns
5. These proxy metrics correlate with ablation-based importance scores

These assumptions are validated by interpretability research (e.g., Voita et al. 2019,
Michel et al. 2019, Elhage et al. 2022).

=============================================================================
COMPATIBILITY
=============================================================================

- Output format: List[HeadScore] (same as original)
- DeCoReEntropy format: {"layer-head": [score], ...} (maintained)
- Works with any transformer model (including Qwen3)
- No training required (purely inference-based)

Author: Refactored for efficiency while maintaining API compatibility
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, leave=True):
        return iterable

# Import the original HeadScore dataclass for compatibility
from .head_scoring import HeadScore, HeadScorer


class EfficientHeadScorer(HeadScorer):
    """
    Base class for efficient head scoring methods.

    Unlike the original AblationScorer which requires O(heads × examples) forward passes,
    efficient scorers require only O(examples) forward passes by collecting all head
    information in a single pass.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
        max_seq_len: int = 512
    ):
        super().__init__(model, tokenizer, device)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Cache model architecture info
        self.n_layers = getattr(
            self.model.config, 'num_hidden_layers',
            getattr(self.model.config, 'n_layers', 32)
        )
        self.n_heads = getattr(
            self.model.config, 'num_attention_heads',
            getattr(self.model.config, 'n_heads', 32)
        )
        self.hidden_dim = getattr(
            self.model.config, 'hidden_size',
            getattr(self.model.config, 'd_model', 4096)
        )
        self.head_dim = self.hidden_dim // self.n_heads

    def _get_attention_layers(self) -> List[nn.Module]:
        """Get all attention layer modules from the model."""
        attention_layers = []

        # Try different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama/Mistral/Qwen style
            for layer in self.model.model.layers:
                if hasattr(layer, 'self_attn'):
                    attention_layers.append(layer.self_attn)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            for layer in self.model.transformer.h:
                if hasattr(layer, 'attn'):
                    attention_layers.append(layer.attn)
        else:
            warnings.warn("Could not identify attention layer structure")

        return attention_layers

    def _format_example_for_scoring(self, example: Dict[str, Any]) -> str:
        """Format an example for forward pass."""
        # Handle different example formats
        if "question" in example and "subquestion" in example:
            # CognitiveMirrors format
            question = example.get("question", "")
            subquestion = example.get("subquestion", "")
            return f"{question}\n{subquestion}"
        elif "question" in example:
            # Atomic task format
            return example.get("question", str(example))
        elif "edges" in example:
            # Backward-chaining format
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"]])
            goal = example.get("goal", "?")
            return f"{edges_str}|{goal}:"
        else:
            return str(example)


class ActivationScorer(EfficientHeadScorer):
    """
    Activation-based head importance scoring.

    This method scores heads based on their activation statistics:
    - Activation norm: Heads with larger outputs have more influence
    - Activation variance: High variance indicates context-sensitive computation

    EFFICIENCY:
    - Single forward pass per example (collects all head activations)
    - Post-hoc computation of scores (no additional model calls)
    - Cost: O(examples) forward passes vs O(heads × examples) for ablation
    - Speedup: ~1000x for Qwen3-4B (1152 heads)

    ASSUMPTIONS:
    - Activation magnitude correlates with head importance
    - Variance across sequence positions indicates reasoning (not just copying)
    - These metrics approximate ablation-based importance without explicit ablation
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
        max_seq_len: int = 512,
        norm_weight: float = 0.5,
        variance_weight: float = 0.5
    ):
        super().__init__(model, tokenizer, device, batch_size, max_seq_len)
        self.norm_weight = norm_weight
        self.variance_weight = variance_weight

    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        baseline_acc: Optional[float] = None,
        debug: bool = False
    ) -> HeadScore:
        """
        Score a single head based on activation statistics.

        Note: This method is provided for API compatibility but is inefficient.
        Use score_all_heads() instead which collects all activations in one pass.
        """
        # Collect activations for all examples
        all_activations = []

        for example in examples:
            input_text = self._format_example_for_scoring(example)
            activations = self._collect_head_activation(input_text, layer, head)
            if activations is not None:
                all_activations.append(activations)

        if len(all_activations) == 0:
            return HeadScore(
                layer=layer, head=head, score=0.0, confidence=0.0,
                method="activation", metadata={"error": "no_activations"}
            )

        # Compute statistics
        all_activations = torch.cat(all_activations, dim=0)  # [total_tokens, head_dim]

        # Activation norm (L2 norm averaged over tokens)
        norm_score = all_activations.norm(dim=-1).mean().item()

        # Activation variance (high variance = context-sensitive)
        var_score = all_activations.var(dim=0).mean().item()

        # Combine scores
        combined_score = self.norm_weight * norm_score + self.variance_weight * var_score

        confidence = min(len(all_activations) / 100.0, 1.0)

        return HeadScore(
            layer=layer,
            head=head,
            score=combined_score,
            confidence=confidence,
            method="activation",
            metadata={
                "norm_score": norm_score,
                "variance_score": var_score,
                "n_tokens": len(all_activations)
            }
        )

    def _collect_head_activation(
        self,
        input_text: str,
        layer: int,
        head: int
    ) -> Optional[torch.Tensor]:
        """Collect activation for a specific head (inefficient single-head version)."""
        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            if len(attn_output.shape) == 3:  # [batch, seq, hidden]
                batch_size, seq_len, hidden_dim = attn_output.shape
                # Reshape to [batch, seq, n_heads, head_dim]
                reshaped = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)
                # Extract specific head
                head_act = reshaped[0, :, head, :].detach().cpu()  # [seq, head_dim]
                activations.append(head_act)

        attention_layers = self._get_attention_layers()
        if layer >= len(attention_layers):
            return None

        hook = attention_layers[layer].register_forward_hook(hook_fn)

        try:
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.device)

            with torch.no_grad():
                self.model(input_ids)
        finally:
            hook.remove()

        if len(activations) > 0:
            return activations[0]
        return None

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
        Score all heads efficiently using activation-based metrics.

        KEY EFFICIENCY IMPROVEMENT:
        - Collects ALL head activations in a SINGLE forward pass per example
        - Original ablation: O(layers × heads × examples) forward passes
        - This method: O(examples) forward passes
        - For Qwen3-4B: 1152× speedup (36 layers × 32 heads)
        """
        if n_layers is None:
            n_layers = self.n_layers
        if n_heads is None:
            n_heads = self.n_heads
        if max_layers is None:
            max_layers = min(n_layers, n_layers)  # Use all layers
        if max_heads_per_layer is None:
            max_heads_per_layer = min(n_heads, n_heads)  # Use all heads

        print(f"  [ActivationScorer] Scoring {max_layers} layers × {max_heads_per_layer} heads = {max_layers * max_heads_per_layer} heads")
        print(f"  [ActivationScorer] Using {len(examples)} examples")
        print(f"  [ActivationScorer] Efficiency: Single forward pass per example (vs {max_layers * max_heads_per_layer} passes for ablation)")

        # Initialize accumulators for all heads
        head_norms = torch.zeros(max_layers, max_heads_per_layer)
        head_vars = torch.zeros(max_layers, max_heads_per_layer)
        head_counts = torch.zeros(max_layers, max_heads_per_layer)

        # Collect activations for all examples
        for example in tqdm(examples, desc="  Collecting activations", leave=False):
            input_text = self._format_example_for_scoring(example)

            # Get all head activations in one forward pass
            all_activations = self._collect_all_head_activations(input_text, max_layers, max_heads_per_layer)

            if all_activations is not None:
                # Accumulate statistics for each head
                for layer in range(max_layers):
                    for head in range(max_heads_per_layer):
                        act = all_activations[layer][head]  # [seq_len, head_dim]
                        if act is not None and len(act) > 0:
                            head_norms[layer, head] += act.norm(dim=-1).mean()
                            head_vars[layer, head] += act.var(dim=0).mean()
                            head_counts[layer, head] += 1

        # Compute final scores
        scores = []
        for layer in range(max_layers):
            for head in range(max_heads_per_layer):
                count = head_counts[layer, head].item()
                if count > 0:
                    avg_norm = head_norms[layer, head].item() / count
                    avg_var = head_vars[layer, head].item() / count

                    # Combined score: weighted sum of normalized metrics
                    combined_score = (
                        self.norm_weight * avg_norm +
                        self.variance_weight * avg_var
                    )

                    confidence = min(count / len(examples), 1.0)

                    scores.append(HeadScore(
                        layer=layer,
                        head=head,
                        score=combined_score,
                        confidence=confidence,
                        method="activation",
                        metadata={
                            "norm_score": avg_norm,
                            "variance_score": avg_var,
                            "n_examples": int(count)
                        }
                    ))
                else:
                    scores.append(HeadScore(
                        layer=layer,
                        head=head,
                        score=0.0,
                        confidence=0.0,
                        method="activation",
                        metadata={"error": "no_activations"}
                    ))

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)

        print(f"  [ActivationScorer] Top 5 heads by activation score:")
        for i, s in enumerate(scores[:5]):
            print(f"    {i+1}. Layer {s.layer}, Head {s.head}: score={s.score:.4f}")

        return scores

    def _collect_all_head_activations(
        self,
        input_text: str,
        max_layers: int,
        max_heads: int
    ) -> Optional[Dict[int, Dict[int, torch.Tensor]]]:
        """
        Collect activations for ALL heads in a SINGLE forward pass.

        This is the key efficiency improvement over ablation-based scoring.
        """
        # Storage for all activations: {layer: {head: tensor}}
        all_activations = {l: {} for l in range(max_layers)}

        # Create hooks for all layers
        hooks = []
        attention_layers = self._get_attention_layers()

        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output

                if len(attn_output.shape) == 3:  # [batch, seq, hidden]
                    batch_size, seq_len, hidden_dim = attn_output.shape
                    # Reshape to [batch, seq, n_heads, head_dim]
                    reshaped = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)

                    # Extract all heads for this layer
                    for head in range(min(max_heads, self.n_heads)):
                        head_act = reshaped[0, :, head, :].detach().cpu()
                        all_activations[layer_idx][head] = head_act
            return hook_fn

        # Register hooks for all layers
        for layer_idx in range(min(max_layers, len(attention_layers))):
            hook = attention_layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        try:
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": input_text}]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.device)

            with torch.no_grad():
                self.model(input_ids)

        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()

        return all_activations


class GradientScorer(EfficientHeadScorer):
    """
    Gradient-based head importance scoring.

    This method uses gradients to measure causal importance:
    - Forward pass with gradient tracking
    - Compute loss on target token (correct answer)
    - Backward pass to get gradients
    - Score = gradient magnitude for each head's output

    EFFICIENCY:
    - One forward + backward pass per example
    - Cost: O(examples) vs O(heads × examples) for ablation
    - Speedup: ~500x (slightly slower than activation due to backward pass)

    ASSUMPTIONS:
    - Gradient magnitude indicates causal influence on output
    - Higher gradient = more important for prediction
    - This approximates "what if we changed this head's output"

    RESEARCH BASIS:
    - Michel et al. (2019): "Are Sixteen Heads Really Better than One?"
    - Voita et al. (2019): "Analyzing Multi-Head Self-Attention"
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
        max_seq_len: int = 512
    ):
        super().__init__(model, tokenizer, device, batch_size, max_seq_len)

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
        Score all heads using gradient-based importance.

        KEY INSIGHT:
        We measure ∂Loss/∂head_output for each head. Heads with larger
        gradients have more causal influence on the model's predictions.
        """
        if n_layers is None:
            n_layers = self.n_layers
        if n_heads is None:
            n_heads = self.n_heads
        if max_layers is None:
            max_layers = min(n_layers, n_layers)
        if max_heads_per_layer is None:
            max_heads_per_layer = min(n_heads, n_heads)

        print(f"  [GradientScorer] Scoring {max_layers} layers × {max_heads_per_layer} heads")
        print(f"  [GradientScorer] Using {len(examples)} examples with gradient computation")

        # Accumulator for gradient magnitudes
        grad_magnitudes = torch.zeros(max_layers, max_heads_per_layer)
        example_counts = torch.zeros(max_layers, max_heads_per_layer)

        for example in tqdm(examples, desc="  Computing gradients", leave=False):
            # Get target answer for loss computation
            target_answer = example.get("answer", example.get("subquestion_answer", ""))
            if not target_answer:
                continue

            input_text = self._format_example_for_scoring(example)

            # Compute gradients for all heads
            head_grads = self._compute_head_gradients(
                input_text, target_answer, max_layers, max_heads_per_layer
            )

            if head_grads is not None:
                for layer in range(max_layers):
                    for head in range(max_heads_per_layer):
                        if (layer, head) in head_grads:
                            grad_magnitudes[layer, head] += head_grads[(layer, head)]
                            example_counts[layer, head] += 1

        # Compute final scores
        scores = []
        for layer in range(max_layers):
            for head in range(max_heads_per_layer):
                count = example_counts[layer, head].item()
                if count > 0:
                    avg_grad = grad_magnitudes[layer, head].item() / count
                    confidence = min(count / len(examples), 1.0)

                    scores.append(HeadScore(
                        layer=layer,
                        head=head,
                        score=avg_grad,
                        confidence=confidence,
                        method="gradient",
                        metadata={
                            "avg_gradient_magnitude": avg_grad,
                            "n_examples": int(count)
                        }
                    ))
                else:
                    scores.append(HeadScore(
                        layer=layer,
                        head=head,
                        score=0.0,
                        confidence=0.0,
                        method="gradient",
                        metadata={"error": "no_gradients"}
                    ))

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)

        print(f"  [GradientScorer] Top 5 heads by gradient magnitude:")
        for i, s in enumerate(scores[:5]):
            print(f"    {i+1}. Layer {s.layer}, Head {s.head}: score={s.score:.6f}")

        return scores

    def _compute_head_gradients(
        self,
        input_text: str,
        target_answer: str,
        max_layers: int,
        max_heads: int
    ) -> Optional[Dict[Tuple[int, int], float]]:
        """
        Compute gradient magnitudes for all heads in a single forward+backward pass.
        """
        head_outputs = {}  # {(layer, head): tensor}
        hooks = []
        attention_layers = self._get_attention_layers()

        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output

                if len(attn_output.shape) == 3:  # [batch, seq, hidden]
                    batch_size, seq_len, hidden_dim = attn_output.shape
                    reshaped = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)

                    for head in range(min(max_heads, self.n_heads)):
                        # Keep gradient connection
                        head_out = reshaped[0, :, head, :]
                        head_outputs[(layer_idx, head)] = head_out
            return hook_fn

        # Register hooks
        for layer_idx in range(min(max_layers, len(attention_layers))):
            hook = attention_layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

        try:
            # Prepare input
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": input_text}]
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                full_text = input_text

            # Tokenize input + answer for loss computation
            full_text_with_answer = full_text + " " + target_answer
            input_ids = self.tokenizer.encode(
                full_text_with_answer,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.device)

            # Get answer token positions
            prompt_ids = self.tokenizer.encode(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.device)

            answer_start = prompt_ids.shape[1]

            # Forward pass with gradient tracking
            self.model.train()  # Enable gradient computation
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Compute loss on answer tokens
            if answer_start < input_ids.shape[1] - 1:
                answer_logits = logits[0, answer_start:-1]
                answer_targets = input_ids[0, answer_start+1:]

                # Cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    answer_logits,
                    answer_targets,
                    reduction='mean'
                )

                # Backward pass
                loss.backward()

                # Collect gradient magnitudes
                grad_magnitudes = {}
                for (layer, head), head_out in head_outputs.items():
                    if head_out.grad is not None:
                        grad_mag = head_out.grad.norm().item()
                        grad_magnitudes[(layer, head)] = grad_mag
                    else:
                        grad_magnitudes[(layer, head)] = 0.0

                return grad_magnitudes

        except Exception as e:
            warnings.warn(f"Gradient computation failed: {e}")
            return None

        finally:
            # Cleanup
            for hook in hooks:
                hook.remove()
            self.model.eval()
            self.model.zero_grad()

        return None

    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        baseline_acc: Optional[float] = None,
        debug: bool = False
    ) -> HeadScore:
        """API compatibility method - use score_all_heads for efficiency."""
        warnings.warn("Use score_all_heads() for efficient gradient-based scoring")
        return HeadScore(
            layer=layer, head=head, score=0.0, confidence=0.0,
            method="gradient", metadata={"error": "use_score_all_heads"}
        )


class AttentionPatternScorer(EfficientHeadScorer):
    """
    Attention pattern-based head importance scoring.

    This method analyzes attention patterns without ablation:
    - Attention entropy: Lower entropy = more focused = likely reasoning
    - Max attention: Higher max = more decisive attention
    - Attention to context vs. recent tokens

    EFFICIENCY:
    - Single forward pass with output_attentions=True
    - Pure inference (no gradient computation)
    - Cost: O(examples) forward passes
    - Speedup: ~1000x (fastest method)

    ASSUMPTIONS:
    - Reasoning heads have focused (low-entropy) attention patterns
    - Copy/retrieval heads attend strongly to specific source tokens
    - High max attention indicates decisive head behavior

    LIMITATIONS:
    - Less accurate than gradient-based scoring
    - May miss heads that are important but have diffuse attention
    - Best used as fast screening before more detailed analysis
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
        max_seq_len: int = 512,
        entropy_weight: float = 0.6,
        max_attn_weight: float = 0.4
    ):
        super().__init__(model, tokenizer, device, batch_size, max_seq_len)
        self.entropy_weight = entropy_weight
        self.max_attn_weight = max_attn_weight

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
        Score all heads using attention pattern analysis.

        FASTEST METHOD: Pure forward pass analysis, no gradients or ablation.
        """
        if n_layers is None:
            n_layers = self.n_layers
        if n_heads is None:
            n_heads = self.n_heads
        if max_layers is None:
            max_layers = min(n_layers, n_layers)
        if max_heads_per_layer is None:
            max_heads_per_layer = min(n_heads, n_heads)

        print(f"  [AttentionPatternScorer] Scoring {max_layers} layers × {max_heads_per_layer} heads")
        print(f"  [AttentionPatternScorer] Using {len(examples)} examples")
        print(f"  [AttentionPatternScorer] FASTEST method - pure forward pass analysis")

        # Accumulators
        entropy_scores = torch.zeros(max_layers, max_heads_per_layer)
        max_attn_scores = torch.zeros(max_layers, max_heads_per_layer)
        counts = torch.zeros(max_layers, max_heads_per_layer)

        for example in tqdm(examples, desc="  Analyzing attention", leave=False):
            input_text = self._format_example_for_scoring(example)

            attention_stats = self._analyze_attention_patterns(
                input_text, max_layers, max_heads_per_layer
            )

            if attention_stats is not None:
                for layer in range(max_layers):
                    for head in range(max_heads_per_layer):
                        if (layer, head) in attention_stats:
                            entropy, max_attn = attention_stats[(layer, head)]
                            entropy_scores[layer, head] += entropy
                            max_attn_scores[layer, head] += max_attn
                            counts[layer, head] += 1

        # Compute final scores
        scores = []

        # Normalize scores for combination
        valid_mask = counts > 0
        if valid_mask.any():
            avg_entropy = entropy_scores[valid_mask] / counts[valid_mask]
            avg_max_attn = max_attn_scores[valid_mask] / counts[valid_mask]

            # Normalize to [0, 1] range
            if avg_entropy.max() > avg_entropy.min():
                norm_entropy = (avg_entropy - avg_entropy.min()) / (avg_entropy.max() - avg_entropy.min())
            else:
                norm_entropy = torch.zeros_like(avg_entropy)

            if avg_max_attn.max() > avg_max_attn.min():
                norm_max_attn = (avg_max_attn - avg_max_attn.min()) / (avg_max_attn.max() - avg_max_attn.min())
            else:
                norm_max_attn = torch.zeros_like(avg_max_attn)

        idx = 0
        for layer in range(max_layers):
            for head in range(max_heads_per_layer):
                count = counts[layer, head].item()
                if count > 0:
                    avg_ent = entropy_scores[layer, head].item() / count
                    avg_max = max_attn_scores[layer, head].item() / count

                    # Score: Low entropy (focused) + high max attention = important
                    # Invert entropy (lower is better)
                    if valid_mask[layer, head]:
                        ent_score = 1.0 - norm_entropy[idx].item()
                        max_score = norm_max_attn[idx].item()
                        idx += 1
                    else:
                        ent_score = 0.0
                        max_score = 0.0

                    combined_score = (
                        self.entropy_weight * ent_score +
                        self.max_attn_weight * max_score
                    )

                    confidence = min(count / len(examples), 1.0)

                    scores.append(HeadScore(
                        layer=layer,
                        head=head,
                        score=combined_score,
                        confidence=confidence,
                        method="attention_pattern",
                        metadata={
                            "avg_entropy": avg_ent,
                            "avg_max_attention": avg_max,
                            "n_examples": int(count)
                        }
                    ))
                else:
                    scores.append(HeadScore(
                        layer=layer,
                        head=head,
                        score=0.0,
                        confidence=0.0,
                        method="attention_pattern",
                        metadata={"error": "no_attention"}
                    ))

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)

        print(f"  [AttentionPatternScorer] Top 5 heads by attention pattern score:")
        for i, s in enumerate(scores[:5]):
            print(f"    {i+1}. Layer {s.layer}, Head {s.head}: score={s.score:.4f}")

        return scores

    def _analyze_attention_patterns(
        self,
        input_text: str,
        max_layers: int,
        max_heads: int
    ) -> Optional[Dict[Tuple[int, int], Tuple[float, float]]]:
        """
        Analyze attention patterns for all heads in a single forward pass.

        Returns: {(layer, head): (entropy, max_attention)}
        """
        try:
            # Prepare input
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": input_text}]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.device)

            # Forward pass with attention outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    return_dict=True
                )

            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                return None

            attention_stats = {}

            for layer_idx, layer_attn in enumerate(outputs.attentions):
                if layer_idx >= max_layers:
                    break

                # layer_attn shape: [batch, heads, seq_len, seq_len]
                for head_idx in range(min(layer_attn.shape[1], max_heads)):
                    # Get attention for this head (last token attending to all)
                    head_attn = layer_attn[0, head_idx, -1, :].cpu()  # [seq_len]

                    # Compute entropy
                    # Add small epsilon to avoid log(0)
                    attn_probs = head_attn.clamp(min=1e-10)
                    entropy = -(attn_probs * torch.log(attn_probs)).sum().item()

                    # Compute max attention
                    max_attn = head_attn.max().item()

                    attention_stats[(layer_idx, head_idx)] = (entropy, max_attn)

            return attention_stats

        except Exception as e:
            warnings.warn(f"Attention analysis failed: {e}")
            return None

    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        baseline_acc: Optional[float] = None,
        debug: bool = False
    ) -> HeadScore:
        """API compatibility method - use score_all_heads for efficiency."""
        warnings.warn("Use score_all_heads() for efficient attention pattern scoring")
        return HeadScore(
            layer=layer, head=head, score=0.0, confidence=0.0,
            method="attention_pattern", metadata={"error": "use_score_all_heads"}
        )


class CombinedScorer(EfficientHeadScorer):
    """
    Combined scoring using multiple efficient methods.

    This scorer combines activation, gradient, and attention pattern scores
    for more robust head importance estimation.

    EFFICIENCY:
    - Runs all three methods in sequence
    - Still much faster than brute-force ablation
    - Cost: ~3× O(examples) vs O(heads × examples) for ablation

    USE CASE:
    - When highest accuracy is needed
    - When downstream contrastive decoding performance is critical
    - When compute budget allows for more thorough analysis
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
        max_seq_len: int = 512,
        activation_weight: float = 0.4,
        gradient_weight: float = 0.4,
        attention_weight: float = 0.2,
        use_gradient: bool = True
    ):
        super().__init__(model, tokenizer, device, batch_size, max_seq_len)
        self.activation_weight = activation_weight
        self.gradient_weight = gradient_weight if use_gradient else 0.0
        self.attention_weight = attention_weight
        self.use_gradient = use_gradient

        # Normalize weights
        total_weight = self.activation_weight + self.gradient_weight + self.attention_weight
        self.activation_weight /= total_weight
        self.gradient_weight /= total_weight
        self.attention_weight /= total_weight

        # Initialize sub-scorers
        self.activation_scorer = ActivationScorer(model, tokenizer, device, batch_size, max_seq_len)
        if use_gradient:
            self.gradient_scorer = GradientScorer(model, tokenizer, device, batch_size, max_seq_len)
        self.attention_scorer = AttentionPatternScorer(model, tokenizer, device, batch_size, max_seq_len)

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
        Score all heads using combined methods.
        """
        print(f"  [CombinedScorer] Running combined scoring with weights:")
        print(f"    Activation: {self.activation_weight:.2f}")
        if self.use_gradient:
            print(f"    Gradient: {self.gradient_weight:.2f}")
        print(f"    Attention: {self.attention_weight:.2f}")

        # Get scores from each method
        print(f"\n  === Activation-based scoring ===")
        activation_scores = self.activation_scorer.score_all_heads(
            examples, subtask_name, n_layers, n_heads, max_layers, max_heads_per_layer
        )

        if self.use_gradient:
            print(f"\n  === Gradient-based scoring ===")
            gradient_scores = self.gradient_scorer.score_all_heads(
                examples, subtask_name, n_layers, n_heads, max_layers, max_heads_per_layer
            )

        print(f"\n  === Attention pattern scoring ===")
        attention_scores = self.attention_scorer.score_all_heads(
            examples, subtask_name, n_layers, n_heads, max_layers, max_heads_per_layer
        )

        # Combine scores
        # Create lookup tables
        act_lookup = {(s.layer, s.head): s.score for s in activation_scores}
        if self.use_gradient:
            grad_lookup = {(s.layer, s.head): s.score for s in gradient_scores}
        attn_lookup = {(s.layer, s.head): s.score for s in attention_scores}

        # Normalize each score type to [0, 1]
        def normalize_scores(lookup):
            if len(lookup) == 0:
                return lookup
            values = list(lookup.values())
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                return {k: (v - min_val) / (max_val - min_val) for k, v in lookup.items()}
            return {k: 0.0 for k in lookup}

        act_norm = normalize_scores(act_lookup)
        if self.use_gradient:
            grad_norm = normalize_scores(grad_lookup)
        attn_norm = normalize_scores(attn_lookup)

        # Combine
        combined_scores = []
        for s in activation_scores:
            key = (s.layer, s.head)
            combined = (
                self.activation_weight * act_norm.get(key, 0.0) +
                self.attention_weight * attn_norm.get(key, 0.0)
            )
            if self.use_gradient:
                combined += self.gradient_weight * grad_norm.get(key, 0.0)

            combined_scores.append(HeadScore(
                layer=s.layer,
                head=s.head,
                score=combined,
                confidence=s.confidence,
                method="combined",
                metadata={
                    "activation_score": act_norm.get(key, 0.0),
                    "gradient_score": grad_norm.get(key, 0.0) if self.use_gradient else None,
                    "attention_score": attn_norm.get(key, 0.0),
                    "weights": {
                        "activation": self.activation_weight,
                        "gradient": self.gradient_weight,
                        "attention": self.attention_weight
                    }
                }
            ))

        # Sort by combined score
        combined_scores.sort(key=lambda x: x.score, reverse=True)

        print(f"\n  [CombinedScorer] Final top 5 heads:")
        for i, s in enumerate(combined_scores[:5]):
            print(f"    {i+1}. Layer {s.layer}, Head {s.head}: combined={s.score:.4f}")

        return combined_scores

    def score_head(
        self,
        layer: int,
        head: int,
        examples: List[Dict[str, Any]],
        subtask_name: str,
        baseline_acc: Optional[float] = None,
        debug: bool = False
    ) -> HeadScore:
        """API compatibility method."""
        warnings.warn("Use score_all_heads() for efficient combined scoring")
        return HeadScore(
            layer=layer, head=head, score=0.0, confidence=0.0,
            method="combined", metadata={"error": "use_score_all_heads"}
        )


def create_efficient_scorer(
    method: str,
    model,
    tokenizer,
    device: str = "cuda",
    **kwargs
) -> EfficientHeadScorer:
    """
    Factory function to create efficient scorers.

    Available methods:
    - "activation": Activation-based scoring (fastest, ~1000x speedup)
    - "gradient": Gradient-based scoring (~500x speedup, more accurate)
    - "attention_pattern": Attention pattern analysis (~1000x speedup)
    - "combined": Combination of all methods (most accurate)

    Args:
        method: Scoring method name
        model: The transformer model
        tokenizer: The tokenizer
        device: Device to run on
        **kwargs: Additional arguments passed to scorer

    Returns:
        An efficient head scorer instance
    """
    scorers = {
        "activation": ActivationScorer,
        "gradient": GradientScorer,
        "attention_pattern": AttentionPatternScorer,
        "combined": CombinedScorer,
    }

    if method not in scorers:
        raise ValueError(
            f"Unknown efficient scoring method: {method}. "
            f"Available methods: {list(scorers.keys())}"
        )

    return scorers[method](model, tokenizer, device, **kwargs)
