"""
Subtask extraction from CognitiveMirrors dataset.

This module discovers subtasks for Logical Reasoning questions.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Subtask:
    """Represents a reasoning subtask."""
    
    name: str
    description: str
    source_file: str
    anchor_line: Optional[int] = None
    anchor_function: Optional[str] = None
    task_type: str = "logical_reasoning"
    
    def __repr__(self):
        return f"Subtask(name={self.name}, type={self.task_type})"
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "source_file": self.source_file,
            "anchor_line": self.anchor_line,
            "anchor_function": self.anchor_function,
            "task_type": self.task_type
        }


def parse_cognitive_mirrors_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a CognitiveMirrors example.
    
    Format: {
        "question": "...",
        "subquestion": "...",
        "subquestion_answer": "...",
        "cognitive_skill": "Logical Reasoning"
    }
    """
    if not example:
        return None
    
    question = example.get("question", "")
    subquestion = example.get("subquestion", "")
    subquestion_answer = example.get("subquestion_answer", "")
    cognitive_skill = example.get("cognitive_skill", "")
    
    if not question or not subquestion or not subquestion_answer:
        return None
    
    if cognitive_skill != "Logical Reasoning":
        return None
    
    return {
        "question": question,
        "subquestion": subquestion,
        "answer": subquestion_answer,
        "cognitive_skill": cognitive_skill,
        "main_answer": example.get("main_answer", ""),
        "full_example": example.get("full_example", {})
    }


def discover_subtasks(cognitive_mirrors_dir: str) -> List[Subtask]:
    """
    Discover subtasks from CognitiveMirrors dataset.
    
    For Logical Reasoning, we focus on the main subtask:
    - logical_reasoning: Answer logical reasoning subquestions
    """
    subtasks = []
    
    # Main Logical Reasoning subtask
    subtasks.append(Subtask(
        name="logical_reasoning",
        description="Answer logical reasoning subquestions based on main question and context",
        source_file=os.path.join(cognitive_mirrors_dir, "dataset", "balanced_cot_train_data.json"),
        task_type="logical_reasoning"
    ))
    
    return subtasks


def get_subtask_examples(
    dataset_file: str,
    subtask: Subtask,
    max_examples: int = 100
) -> List[Dict[str, Any]]:
    """
    Get examples for a specific subtask from the CognitiveMirrors dataset.
    """
    if not os.path.exists(dataset_file):
        # Try preprocessed file
        preprocessed_file = "cognitive_mirrors_logical_reasoning.json"
        if os.path.exists(preprocessed_file):
            dataset_file = preprocessed_file
        else:
            return []
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    examples = []
    for item in data:
        parsed = parse_cognitive_mirrors_example(item)
        if parsed:
            examples.append(parsed)
            if len(examples) >= max_examples:
                break
    
    return examples

