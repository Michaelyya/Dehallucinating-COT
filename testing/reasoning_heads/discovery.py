"""
Main discovery framework for identifying reasoning heads.
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from .subtask_extraction import discover_subtasks, Subtask, get_subtask_examples
from .head_scoring import HeadScorer, HeadScore, create_scorer


@dataclass
class ReasoningHead:
    """Identified reasoning head with metadata."""
    layer: int
    head: int
    subtask: str
    score: float
    confidence: float
    method: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            "layer": self.layer,
            "head": self.head,
            "subtask": self.subtask,
            "score": self.score,
            "confidence": self.confidence,
            "method": self.method,
            "metadata": self.metadata or {}
        }


class ReasoningHeadDiscovery:
    """
    Main class for discovering reasoning heads.
    
    Discovers subtasks, collects head traces, and scores heads.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        backward_chaining_dir: str = "../backward-chaining-circuits",
        device: str = "cuda",
        scoring_method: str = "ablation",
        scoring_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.backward_chaining_dir = backward_chaining_dir
        self.scoring_method = scoring_method
        self.scoring_config = scoring_config or {}
        self.cache_dir = cache_dir
        
        # Set cache directory if provided
        if cache_dir:
            import os
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_DATASETS_CACHE"] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize scorer
        self.scorer = create_scorer(
            scoring_method,
            model,
            tokenizer,
            device,
            **self.scoring_config
        )
        
        # Discover subtasks
        self.subtasks = discover_subtasks(backward_chaining_dir)
        print(f"Discovered {len(self.subtasks)} subtasks")
    
    def discover_heads(
        self,
        dataset_file: Optional[str] = None,
        n_examples_per_subtask: int = 20,
        top_k: int = 10,
        min_score: float = 0.1,
        min_confidence: float = 0.3
    ) -> List[ReasoningHead]:
        """
        Discover reasoning heads for all subtasks.
        
        Args:
            dataset_file: Path to dataset file
            n_examples_per_subtask: Number of examples to use per subtask
            top_k: Return top K heads per subtask
            min_score: Minimum score threshold
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of identified reasoning heads
        """
        if dataset_file is None:
            dataset_file = os.path.join(self.backward_chaining_dir, "dataset.txt")
        
        all_heads = []
        
        for subtask in self.subtasks:
            print(f"\nDiscovering heads for subtask: {subtask.name}")
            
            # Get examples for this subtask
            examples = get_subtask_examples(
                subtask,
                dataset_file,
                max_examples=n_examples_per_subtask
            )
            
            if len(examples) == 0:
                print(f"  Warning: No examples found for {subtask.name}")
                continue
            
            print(f"  Using {len(examples)} examples")
            
            # Score all heads for this subtask
            head_scores = self.scorer.score_all_heads(
                examples,
                subtask.name
            )
            
            # Filter and select top heads
            filtered_scores = [
                score for score in head_scores
                if score.score >= min_score and score.confidence >= min_confidence
            ]
            
            # Take top K
            top_scores = filtered_scores[:top_k]
            
            # Convert to ReasoningHead objects
            for score in top_scores:
                reasoning_head = ReasoningHead(
                    layer=score.layer,
                    head=score.head,
                    subtask=subtask.name,
                    score=score.score,
                    confidence=score.confidence,
                    method=score.method,
                    metadata=score.metadata
                )
                all_heads.append(reasoning_head)
            
            print(f"  Found {len(top_scores)} reasoning heads")
        
        return all_heads
    
    def collect_head_traces(
        self,
        examples: List[Dict[str, Any]],
        subtask: Subtask,
        output_dir: str = "./traces"
    ) -> Dict[str, Any]:
        """
        Collect per-layer, per-head attention maps and activations.
        
        Args:
            examples: List of examples to trace
            subtask: Subtask being analyzed
            output_dir: Directory to save traces
        
        Returns:
            Dictionary with trace data and metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        traces = {
            "subtask": subtask.to_dict(),
            "n_examples": len(examples),
            "attention_maps": [],
            "activations": [],
            "metadata": {}
        }
        
        # Collect traces for each example
        for i, example in enumerate(examples):
            trace = self._trace_example(example, subtask)
            if trace:
                traces["attention_maps"].append(trace["attention"])
                traces["activations"].append(trace["activations"])
        
        # Save traces
        trace_file = os.path.join(output_dir, f"traces_{subtask.name}.json")
        with open(trace_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_traces = self._serialize_traces(traces)
            json.dump(json_traces, f, indent=2)
        
        print(f"Saved traces to {trace_file}")
        return traces
    
    def _trace_example(
        self,
        example: Dict[str, Any],
        subtask: Subtask
    ) -> Optional[Dict[str, Any]]:
        """Trace a single example through the model."""
        try:
            # Format input
            input_text = self._format_example(example)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Run model with hooks to collect activations
            # This is simplified - actual implementation needs proper hooking
            with torch.no_grad():
                # Try to get attention if model supports it
                if hasattr(self.model, 'generate'):
                    outputs = self.model(
                        input_ids,
                        output_attentions=True,
                        return_dict=True
                    )
                    
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        # Extract attention maps
                        attention = []
                        for layer_attn in outputs.attentions:
                            # Shape: [batch, heads, seq_len, seq_len]
                            attention.append(layer_attn[0].cpu().numpy())
                        
                        return {
                            "attention": attention,
                            "activations": {}  # Would need to hook for activations
                        }
            
            return None
        except Exception as e:
            print(f"Warning: Could not trace example: {e}")
            return None
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example for model input."""
        if "edges" in example:
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"]])
            goal = example.get("goal", "?")
            return f"{edges_str}|{goal}:"
        return str(example)
    
    def _serialize_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize traces for JSON (convert numpy arrays)."""
        serialized = {}
        for key, value in traces.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    serialized[key] = [arr.tolist() for arr in value]
                else:
                    serialized[key] = value
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def save_discovered_heads(
        self,
        heads: List[ReasoningHead],
        output_file: str = "discovered_heads.json"
    ):
        """Save discovered heads to JSON file."""
        data = {
            "model": getattr(self.model.config, 'model_type', 'unknown'),
            "scoring_method": self.scoring_method,
            "n_subtasks": len(self.subtasks),
            "heads": [head.to_dict() for head in heads]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(heads)} discovered heads to {output_file}")
    
    def load_discovered_heads(
        self,
        input_file: str
    ) -> List[ReasoningHead]:
        """Load discovered heads from JSON file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        heads = []
        for head_dict in data.get("heads", []):
            heads.append(ReasoningHead(**head_dict))
        
        return heads
    
    def get_heads_for_subtask(
        self,
        heads: List[ReasoningHead],
        subtask_name: str
    ) -> List[ReasoningHead]:
        """Get all heads for a specific subtask."""
        return [h for h in heads if h.subtask == subtask_name]
    
    def get_head_list_for_masking(
        self,
        heads: List[ReasoningHead],
        subtask_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Get list of (layer, head) tuples for masking.
        
        Format compatible with DeCoRe block_list parameter.
        """
        if subtask_name:
            heads = self.get_heads_for_subtask(heads, subtask_name)
        
        # Sort by score
        heads = sorted(heads, key=lambda x: x.score, reverse=True)
        
        if top_k:
            heads = heads[:top_k]
        
        return [(h.layer, h.head) for h in heads]

