"""
Preprocess CognitiveMirrors dataset to filter for Logical Reasoning questions.
"""

import json
import os
from typing import List, Dict, Any


def filter_logical_reasoning(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter dataset to keep only questions with Logical Reasoning cognitive_skill.
    
    Args:
        data: List of examples from balanced_cot_train_data.json
        
    Returns:
        Filtered list with only Logical Reasoning questions
    """
    filtered = []
    
    for example in data:
        # Check if this example has any subquestion with Logical Reasoning
        has_logical_reasoning = False
        logical_reasoning_subquestion = None
        
        if "generated" in example:
            for subq in example["generated"]:
                if subq.get("cognitive_skill") == "Logical Reasoning":
                    has_logical_reasoning = True
                    logical_reasoning_subquestion = subq
                    break
        
        # Also check if answer matches the expected format
        if has_logical_reasoning and logical_reasoning_subquestion:
            answer = logical_reasoning_subquestion.get("answer", "")
            # Check if answer matches expected format (e.g., "No, the statement is not factual.")
            if answer and len(answer.strip()) > 0:
                filtered.append({
                    "question": example.get("question", ""),
                    "main_answer": example.get("answer", ""),
                    "subquestion": logical_reasoning_subquestion.get("subquestion", ""),
                    "subquestion_answer": logical_reasoning_subquestion.get("answer", ""),
                    "cognitive_skill": "Logical Reasoning",
                    "full_example": example  # Keep full example for reference
                })
    
    return filtered


def main():
    """Main preprocessing function."""
    # Paths
    input_file = "../../CognitiveMirrors/dataset/balanced_cot_train_data.json"
    output_file = "cognitive_mirrors_logical_reasoning.json"
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)} examples")
    
    # Filter for Logical Reasoning
    filtered_data = filter_logical_reasoning(data)
    
    print(f"Filtered dataset size: {len(filtered_data)} Logical Reasoning examples")
    
    # Save filtered dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved filtered dataset to {output_file}")
    
    # Print some statistics
    if filtered_data:
        print("\nSample examples:")
        for i, example in enumerate(filtered_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"  Question: {example['question'][:100]}...")
            print(f"  Subquestion: {example['subquestion']}")
            print(f"  Answer: {example['subquestion_answer']}")


if __name__ == "__main__":
    main()

