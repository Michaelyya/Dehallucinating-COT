import os
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Try to import CognitiveMirrors version first, fallback to backward-chaining
try:
    from .subtask_extraction_cognitive_mirrors import (
        discover_subtasks, 
        Subtask, 
        get_subtask_examples,
        parse_cognitive_mirrors_example as parse_example
    )
    USE_COGNITIVE_MIRRORS = True
except ImportError:
    from .subtask_extraction import (
        discover_subtasks, 
        Subtask, 
        get_subtask_examples,
        parse_backward_chaining_example as parse_example
    )
    USE_COGNITIVE_MIRRORS = False
from .head_scoring import HeadScorer, HeadScore, create_scorer

# Try to import efficient scorers
try:
    from .efficient_head_scoring import create_efficient_scorer
    EFFICIENT_SCORERS_AVAILABLE = True
except ImportError:
    EFFICIENT_SCORERS_AVAILABLE = False
    create_efficient_scorer = None


@dataclass
class ReasoningHead:
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
    
    def __init__(
        self,
        model,
        tokenizer,
        backward_chaining_dir: str = "../CognitiveMirrors",
        device: str = "cuda",
        scoring_method: str = "ablation",
        scoring_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Reasoning Head Discovery system.

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            backward_chaining_dir: Directory containing datasets
            device: Device to run on ("cuda" or "cpu")
            scoring_method: Method for scoring heads. Options:
                - "ablation": Original brute-force method (slow)
                - "activation": Activation-based scoring (~1000x faster)
                - "gradient": Gradient-based scoring (~500x faster)
                - "attention_pattern": Attention pattern analysis (~1000x faster)
                - "combined": Multi-method combination (~300x faster)
            scoring_config: Additional configuration for the scorer
            cache_dir: Cache directory for models
        """
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

        # Determine if using efficient scorer
        efficient_methods = ["activation", "gradient", "attention_pattern", "combined"]
        self.use_efficient_scorer = scoring_method in efficient_methods

        if self.use_efficient_scorer:
            if not EFFICIENT_SCORERS_AVAILABLE:
                raise ImportError(
                    f"Efficient scoring method '{scoring_method}' requires efficient_head_scoring module. "
                    "Please ensure efficient_head_scoring.py is available."
                )
            print(f"[ReasoningHeadDiscovery] Using EFFICIENT scorer: {scoring_method}")
            print(f"  Expected speedup: ~{{'activation': 1000, 'gradient': 500, 'attention_pattern': 1000, 'combined': 300}[scoring_method]}x over ablation")
            self.scorer = create_efficient_scorer(
                scoring_method,
                model,
                tokenizer,
                device,
                **self.scoring_config
            )
        else:
            print(f"[ReasoningHeadDiscovery] Using standard scorer: {scoring_method}")
            # Initialize scorer using original method
            self.scorer = create_scorer(
                scoring_method,
                model,
                tokenizer,
                device,
                **self.scoring_config
            )
        
        # Resolve backward_chaining_dir path
        if not os.path.isabs(backward_chaining_dir):
            # Try relative to current working directory
            abs_path = os.path.abspath(backward_chaining_dir)
            if os.path.exists(abs_path):
                backward_chaining_dir = abs_path
            else:
                # Try relative to this file's directory
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                abs_path = os.path.join(base_dir, backward_chaining_dir.lstrip("./"))
                if os.path.exists(abs_path):
                    backward_chaining_dir = abs_path
        
        self.backward_chaining_dir = backward_chaining_dir
        self.use_cognitive_mirrors = USE_COGNITIVE_MIRRORS
        
        if self.use_cognitive_mirrors:
            print(f"Using CognitiveMirrors directory: {self.backward_chaining_dir}")
        else:
            print(f"Using backward-chaining directory: {self.backward_chaining_dir}")
        
        # Discover subtasks
        self.subtasks = discover_subtasks(backward_chaining_dir)
        print(f"Discovered {len(self.subtasks)} subtasks")
    
    def discover_heads(
        self,
        dataset_file: Optional[str] = None,
        n_examples_per_subtask: int = 20,
        top_k: int = 10,
        min_score: float = 0.0,  # Lowered default threshold
        min_confidence: float = 0.0,  # Lowered default threshold
        single_subtask: Optional[str] = None  # Run only one subtask
    ) -> List[ReasoningHead]:
        if dataset_file is None:
            if self.use_cognitive_mirrors:
                # Use preprocessed CognitiveMirrors dataset - try multiple locations
                possible_paths = [
                    "cognitive_mirrors_logical_reasoning.json",  # Current directory
                    os.path.join(os.path.dirname(__file__), "cognitive_mirrors_logical_reasoning.json"),  # Same dir as script
                    os.path.join(self.backward_chaining_dir, "dataset", "balanced_cot_train_data.json"),  # Original dataset
                ]
                dataset_file = None
                for path in possible_paths:
                    if os.path.exists(path):
                        dataset_file = path
                        break
                if dataset_file is None:
                    print("ERROR: Could not find cognitive_mirrors_logical_reasoning.json")
                    print("Please run: python preprocess_cognitive_mirrors.py")
                    return []
            else:
                dataset_file = os.path.join(self.backward_chaining_dir, "dataset.txt")
        
        # Resolve absolute path
        if not os.path.isabs(dataset_file):
            # Try relative to backward_chaining_dir first
            abs_path = os.path.join(self.backward_chaining_dir, dataset_file)
            if os.path.exists(abs_path):
                dataset_file = abs_path
            else:
                # Try relative to current working directory
                abs_path = os.path.abspath(dataset_file)
                if os.path.exists(abs_path):
                    dataset_file = abs_path
        
        print(f"Using dataset file: {dataset_file}")
        if not os.path.exists(dataset_file):
            print(f"ERROR: Dataset file not found at {dataset_file}")
            print(f"Please check the path. Backward-chaining dir: {self.backward_chaining_dir}")
            return []
        
        all_heads = []
        
        # First, try to load some examples to verify the dataset works
        print(f"\nLoading examples from dataset...")
        all_examples = []
        try:
            if self.use_cognitive_mirrors:
                # Load JSON file for CognitiveMirrors
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data[:n_examples_per_subtask * 2]:  # Load a few extra
                        parsed = parse_example(item)
                        if parsed:
                            all_examples.append(parsed)
            else:
                # Load line-by-line for backward-chaining
                with open(dataset_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= n_examples_per_subtask * 2:  # Load a few extra
                            break
                        parsed = parse_example(line.strip())
                        if parsed:
                            all_examples.append(parsed)
            print(f"Successfully loaded {len(all_examples)} examples from dataset")
        except Exception as e:
            print(f"ERROR: Could not load dataset: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        if len(all_examples) == 0:
            print("ERROR: No valid examples found in dataset file")
            return []
        
        # Filter subtasks if single_subtask is specified
        subtasks_to_process = self.subtasks
        if single_subtask:
            subtasks_to_process = [s for s in self.subtasks if s.name == single_subtask]
            if len(subtasks_to_process) == 0:
                print(f"ERROR: Subtask '{single_subtask}' not found. Available subtasks:")
                for s in self.subtasks:
                    print(f"  - {s.name}")
                return []
            print(f"\nProcessing only subtask: {single_subtask}")
        
        for subtask in subtasks_to_process:
            print(f"\n{'='*60}")
            print(f"Discovering heads for subtask: {subtask.name}")
            print(f"Description: {subtask.description}")
            print(f"Type: {subtask.task_type}")
            print(f"{'='*60}")
            
            # Use all examples for now (they're all relevant to backward-chaining)
            # In the future, we can filter by subtask type
            examples = all_examples[:n_examples_per_subtask]
            
            if len(examples) == 0:
                print(f"  Warning: No examples available for {subtask.name}")
                continue
            
            print(f"\n  Using {len(examples)} examples")
            print(f"  Example format: {self._format_example_for_display(examples[0])}")
            
            # Score all heads for this subtask
            print(f"\n  Scoring heads...")
            head_scores = self.scorer.score_all_heads(
                examples,
                subtask.name
            )
            
            print(f"\n  Scoring results:")
            print(f"    Total heads scored: {len(head_scores)}")
            if len(head_scores) > 0:
                print(f"    Score range: {min(s.score for s in head_scores):.4f} to {max(s.score for s in head_scores):.4f}")
                print(f"    Top 5 scores:")
                for i, score in enumerate(head_scores[:5]):
                    print(f"      {i+1}. Layer {score.layer}, Head {score.head}: score={score.score:.4f}, confidence={score.confidence:.4f}")
            
            # Filter and select top heads
            # For CognitiveMirrors, we want heads with LARGEST positive scores
            # (heads that decrease performance the most when masked)
            filtered_scores = [
                score for score in head_scores
                if score.score >= min_score and score.confidence >= min_confidence
            ]
            
            print(f"\n  After filtering (min_score={min_score}, min_confidence={min_confidence}):")
            print(f"    {len(filtered_scores)} heads passed threshold")
            
            # Sort by score descending (largest positive scores first)
            # Positive score = baseline > ablated (masking hurts performance)
            filtered_scores.sort(key=lambda x: x.score, reverse=True)
            
            # Take top K (heads with largest performance drop when masked)
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
            
            print(f"\n  Final result: Found {len(top_scores)} reasoning heads")
            
            # If single subtask mode, break after first one
            if single_subtask:
                break
        
        return all_heads
    
    def collect_head_traces(
        self,
        examples: List[Dict[str, Any]],
        subtask: Subtask,
        output_dir: str = "./traces"
    ) -> Dict[str, Any]:
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
        if "edges" in example:
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"]])
            goal = example.get("goal", "?")
            return f"{edges_str}|{goal}:"
        return str(example)
    
    def _format_example_for_display(self, example: Dict[str, Any]) -> str:
        if "edges" in example:
            edges_str = ",".join([f"{e[0]}>{e[1]}" for e in example["edges"][:5]])  # Show first 5
            if len(example["edges"]) > 5:
                edges_str += f"... ({len(example['edges'])} total)"
            goal = example.get("goal", "?")
            path = example.get("path", [])
            path_str = ">".join([str(p) for p in path[:5]])  # Show first 5
            if len(path) > 5:
                path_str += f"... ({len(path)} total)"
            return f"Edges: {edges_str} | Goal: {goal} | Path: {path_str}"
        return str(example)
    
    def _serialize_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
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
        data = {
            "model": getattr(self.model.config, 'model_type', 'unknown'),
            "scoring_method": self.scoring_method,
            "n_subtasks": len(self.subtasks),
            "heads": [head.to_dict() for head in heads]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(heads)} discovered heads to {output_file}")
    
    def save_heads_for_decore(
        self, 
        heads: List[ReasoningHead], 
        output_dir: str = "../retrieval_heads/",
        model_name: str = "Meta-Llama-3-8B-Instruct"
    ):
        """
        Save discovered heads in DeCoReEntropy format.
        
        Format: {"layer-head": [score], ...}
        Example: {"0-5": [0.123], "1-3": [0.456], ...}
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DeCoRe format: {"layer-head": [score], ...}
        decore_format = {}
        for head in heads:
            key = f"{head.layer}-{head.head}"
            # Store score as a list (DeCoRe expects list of scores)
            decore_format[key] = [head.score]
        
        # Sort by score (descending) - DeCoRe will take top K
        sorted_heads = sorted(decore_format.items(), key=lambda x: x[1][0], reverse=True)
        decore_format = dict(sorted_heads)
        
        # Save to file
        output_file = os.path.join(output_dir, f"{model_name}.json")
        with open(output_file, 'w') as f:
            # Write as single line (DeCoRe reads with readline())
            f.write(json.dumps(decore_format))
        
        print(f"Saved {len(decore_format)} heads for DeCoReEntropy to {output_file}")
        print(f"  Format: layer-head -> [score]")
        print(f"  Top 5 heads: {list(sorted_heads[:5])}")
        
        return output_file
    
    def load_discovered_heads(
        self,
        input_file: str
    ) -> List[ReasoningHead]:
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
        return [h for h in heads if h.subtask == subtask_name]
    
    def get_head_list_for_masking(
        self,
        heads: List[ReasoningHead],
        subtask_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        if subtask_name:
            heads = self.get_heads_for_subtask(heads, subtask_name)
        
        # Sort by score
        heads = sorted(heads, key=lambda x: x.score, reverse=True)
        
        if top_k:
            heads = heads[:top_k]
        
        return [(h.layer, h.head) for h in heads]

