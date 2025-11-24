"""
Subtask extraction from backward-chaining-circuits dataset and code.

This module discovers subtasks involved in backward-chaining reasoning
by analyzing the dataset structure and task definitions.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx


class Subtask:
    """Represents a reasoning subtask."""
    
    def __init__(
        self,
        name: str,
        description: str,
        source_file: str,
        anchor_line: Optional[int] = None,
        anchor_function: Optional[str] = None,
        task_type: str = "reasoning_step"
    ):
        self.name = name
        self.description = description
        self.source_file = source_file
        self.anchor_line = anchor_line
        self.anchor_function = anchor_function
        self.task_type = task_type
    
    def __repr__(self):
        return f"Subtask(name={self.name}, type={self.task_type}, file={self.source_file})"
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "source_file": self.source_file,
            "anchor_line": self.anchor_line,
            "anchor_function": self.anchor_function,
            "task_type": self.task_type
        }


def parse_backward_chaining_example(example_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse a backward-chaining example string.
    
    Format: "edge1,edge2,...|goal:path1>path2>..."
    """
    if not example_str or not example_str.strip():
        return None
    
    example_str = example_str.strip()
    parts = example_str.split("|")
    if len(parts) != 2:
        return None
    
    edges_str = parts[0].strip()
    goal_path_str = parts[1].strip()
    
    if not edges_str or not goal_path_str:
        return None
    
    # Parse edges
    edges = []
    for edge_str in edges_str.split(","):
        edge_str = edge_str.strip()
        if ">" in edge_str:
            try:
                out_node, in_node = edge_str.split(">")
                edges.append((int(out_node.strip()), int(in_node.strip())))
            except (ValueError, AttributeError):
                continue
    
    if len(edges) == 0:
        return None
    
    # Parse goal and path
    goal_path_parts = goal_path_str.split(":")
    if len(goal_path_parts) != 2:
        return None
    
    try:
        goal = int(goal_path_parts[0].strip())
        path_str = goal_path_parts[1].strip()
        path = [int(p.strip()) for p in path_str.split(">") if p.strip()]
    except (ValueError, AttributeError):
        return None
    
    try:
        graph = _build_graph(edges)
    except Exception:
        graph = None
    
    return {
        "edges": edges,
        "goal": goal,
        "path": path,
        "graph": graph,
        "source_nodes": set([e[0] for e in edges]),
        "target_nodes": set([e[1] for e in edges]),
        "raw_string": example_str  # Keep original for debugging
    }


def _build_graph(edges: List[Tuple[int, int]]) -> nx.DiGraph:
    """Build a NetworkX graph from edges."""
    G = nx.DiGraph()
    for out_node, in_node in edges:
        G.add_edge(out_node, in_node)
    return G


def discover_subtasks(
    backward_chaining_dir: str = "../backward-chaining-circuits",
    dataset_file: Optional[str] = None
) -> List[Subtask]:
    """
    Discover subtasks from backward-chaining-circuits repository.
    
    Args:
        backward_chaining_dir: Path to backward-chaining-circuits directory
        dataset_file: Optional path to dataset file for analysis
    
    Returns:
        List of discovered subtasks
    """
    subtasks = []
    
    # Resolve paths
    if not os.path.isabs(backward_chaining_dir):
        # Try relative to testing directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backward_chaining_dir = os.path.join(base_dir, backward_chaining_dir)
    
    if dataset_file is None:
        dataset_file = os.path.join(backward_chaining_dir, "dataset.txt")
    
    # Analyze dataset to understand task structure
    if os.path.exists(dataset_file):
        subtasks.extend(_discover_subtasks_from_dataset(dataset_file, backward_chaining_dir))
    
    # Analyze code to find subtask definitions
    src_dir = os.path.join(backward_chaining_dir, "src")
    if os.path.exists(src_dir):
        subtasks.extend(_discover_subtasks_from_code(src_dir))
    
    # Add common reasoning subtasks based on backward-chaining structure
    subtasks.extend(_get_common_backward_chaining_subtasks(backward_chaining_dir))
    
    return subtasks


def _discover_subtasks_from_dataset(
    dataset_file: str,
    base_dir: str
) -> List[Subtask]:
    """Discover subtasks by analyzing dataset examples."""
    subtasks = []
    
    try:
        with open(dataset_file, 'r') as f:
            examples = f.readlines()[:100]  # Sample first 100 examples
        
        # Analyze examples to identify patterns
        edge_parsing_count = 0
        path_finding_count = 0
        goal_identification_count = 0
        node_traversal_count = 0
        
        for example in examples:
            parsed = parse_backward_chaining_example(example.strip())
            if parsed:
                edge_parsing_count += 1
                if parsed["path"]:
                    path_finding_count += 1
                if parsed["goal"] is not None:
                    goal_identification_count += 1
                if len(parsed["path"]) > 1:
                    node_traversal_count += 1
        
        # Create subtasks based on analysis
        if edge_parsing_count > 0:
            subtasks.append(Subtask(
                name="edge_parsing",
                description="Parse edge tokens from input sequence and construct graph representation",
                source_file=dataset_file,
                task_type="parsing"
            ))
        
        if goal_identification_count > 0:
            subtasks.append(Subtask(
                name="goal_identification",
                description="Identify goal node from input and extract target destination",
                source_file=dataset_file,
                task_type="identification"
            ))
        
        if path_finding_count > 0:
            subtasks.append(Subtask(
                name="path_finding",
                description="Find path from root to goal node using backward-chaining",
                source_file=dataset_file,
                task_type="reasoning"
            ))
        
        if node_traversal_count > 0:
            subtasks.append(Subtask(
                name="node_traversal",
                description="Traverse graph nodes step-by-step in backward direction",
                source_file=dataset_file,
                task_type="reasoning"
            ))
    
    except Exception as e:
        print(f"Warning: Could not analyze dataset file: {e}")
    
    return subtasks


def _discover_subtasks_from_code(src_dir: str) -> List[Subtask]:
    """Discover subtasks by analyzing source code."""
    subtasks = []
    
    # Key files to analyze
    key_files = {
        "utils.py": ["is_model_correct", "eval_model", "extract_adj_matrix"],
        "attention_knockout.py": ["attention_knockout_discovery"],
        "interp_utils.py": ["activation_patching", "aggregate_activations"],
    }
    
    for filename, functions in key_files.items():
        filepath = os.path.join(src_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    for func_name in functions:
                        if f"def {func_name}" in line:
                            # Extract function context
                            description = _extract_function_description(lines, i)
                            subtasks.append(Subtask(
                                name=func_name,
                                description=description,
                                source_file=filepath,
                                anchor_line=i + 1,
                                anchor_function=func_name,
                                task_type="computation"
                            ))
    
    return subtasks


def _extract_function_description(lines: List[str], start_line: int) -> str:
    """Extract docstring or infer description from function."""
    # Look for docstring
    for i in range(start_line + 1, min(start_line + 10, len(lines))):
        line = lines[i].strip()
        if '"""' in line or "'''" in line:
            # Extract docstring
            docstring_lines = []
            quote_char = '"""' if '"""' in line else "'''"
            if line.count(quote_char) == 2:
                # Single line docstring
                docstring = line.split(quote_char)[1]
                return docstring.strip()
            else:
                # Multi-line docstring
                for j in range(i + 1, min(i + 20, len(lines))):
                    docstring_lines.append(lines[j])
                    if quote_char in lines[j]:
                        break
                return " ".join(docstring_lines).replace(quote_char, "").strip()
    
    # Fallback: use function name
    func_line = lines[start_line]
    match = re.search(r'def\s+(\w+)', func_line)
    if match:
        return f"Function: {match.group(1)}"
    return "No description available"


def _get_common_backward_chaining_subtasks(base_dir: str) -> List[Subtask]:
    """Get common backward-chaining subtasks based on known structure."""
    return [
        Subtask(
            name="graph_construction",
            description="Construct graph representation from edge tokens",
            source_file=os.path.join(base_dir, "src", "utils.py"),
            anchor_function="extract_adj_matrix",
            task_type="representation"
        ),
        Subtask(
            name="backward_chain_step",
            description="Execute one step of backward-chaining reasoning",
            source_file=os.path.join(base_dir, "src", "utils.py"),
            anchor_function="eval_model",
            task_type="reasoning"
        ),
        Subtask(
            name="path_validation",
            description="Validate that generated path is correct",
            source_file=os.path.join(base_dir, "src", "utils.py"),
            anchor_function="is_model_correct",
            task_type="validation"
        ),
        Subtask(
            name="token_prediction",
            description="Predict next token in backward-chaining sequence",
            source_file=os.path.join(base_dir, "training.py"),
            task_type="generation"
        ),
    ]


def get_subtask_examples(
    subtask: Subtask,
    dataset_file: str,
    max_examples: int = 10
) -> List[Dict[str, Any]]:
    """
    Get example instances for a specific subtask.
    
    Args:
        subtask: The subtask to get examples for
        dataset_file: Path to dataset file
        max_examples: Maximum number of examples to return
    
    Returns:
        List of example dictionaries
    """
    examples = []
    
    if not os.path.exists(dataset_file):
        return examples
    
    try:
        with open(dataset_file, 'r') as f:
            for line in f:
                if len(examples) >= max_examples:
                    break
                
                parsed = parse_backward_chaining_example(line.strip())
                if parsed:
                    # Filter examples relevant to this subtask
                    if _is_relevant_to_subtask(parsed, subtask):
                        examples.append(parsed)
    
    except Exception as e:
        print(f"Warning: Could not load examples: {e}")
    
    return examples


def _is_relevant_to_subtask(example: Dict[str, Any], subtask: Subtask) -> bool:
    """Check if example is relevant to subtask."""
    # For now, all valid examples are relevant
    # Can be refined based on subtask type
    return True

