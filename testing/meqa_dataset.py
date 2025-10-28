"""
MEQA Dataset for Multi-hop Event-centric Question Answering
"""

import json
import os
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MEQADataset(Dataset):
    """
    Dataset class for MEQA (Multi-hop Event-centric Question Answering)
    
    Data format:
    {
        "example_id": "dev_0_s1_3",
        "context": "Roadside IED kills Russian major general [...]",
        "question": "Who died before AI-monitor reported it online?",
        "answer": "major general,local commander,lieutenant general",
        "explanation": [
            "What event contains Al-Monitor is the communicator? reported",
            "What event is after #1 has a victim? killed",
            "Who died in the #2? major general,local commander,lieutenant general"
        ]
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_context_length: int = 4096,
        max_question_length: int = 512,
        max_explanation_length: int = 1024,
        use_chat_template: bool = True,
        num_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.max_explanation_length = max_explanation_length
        self.use_chat_template = use_chat_template
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Limit number of samples if specified
        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        print(f"Loaded {len(self.data)} MEQA examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract components
        example_id = item["example_id"]
        context = item["context"]
        question = item["question"]
        answer = item["answer"]
        explanation = item["explanation"]
        
        # Create the full prompt for CoT reasoning
        if self.use_chat_template:
            # Format as a conversation for instruct models
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the given context. Provide a concise answer."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]
            
            # Return the text string, not tokenized - let DeCoRe framework handle tokenization
            prompted_question = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,  # Don't tokenize, return text
            )
        else:
            # Simple format for base models
            prompted_question = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Create reference explanation (for evaluation)
        reference_explanation = " ".join(explanation)
        
        return {
            "idx": idx,
            "example_id": example_id,
            "prompted_question": prompted_question,
            "context": context,
            "question": question,
            "answer": answer,
            "explanation": explanation,
            "reference_explanation": reference_explanation,
            # Additional fields for compatibility with existing framework
            "verbalised_instruction": [""],  # No separate instruction
            "verbalised_icl_demo": [""],    # No in-context examples
            "verbalised_contexts": [context],
            "verbalised_question": [question],
            "verbalised_answer_prefix": [""],
        }
    
    def collate_fn(self, batch):
        """Custom collate function for MEQA data"""
        # Extract the first item to get the structure
        first_item = batch[0]
        
        # Collate all fields
        collated = {}
        for key in first_item.keys():
            # Handle all fields as lists - DeCoRe framework will handle tokenization
            collated[key] = [item[key] for item in batch]
        
        return collated


def load_meqa_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "dev",
    **kwargs
) -> MEQADataset:
    """
    Load MEQA dataset from the specified path
    
    Args:
        data_path: Path to the MEQA data directory
        tokenizer: Tokenizer to use
        split: Dataset split ('train', 'dev', 'test')
        **kwargs: Additional arguments for MEQADataset
    
    Returns:
        MEQADataset instance
    """
    # Construct the full path to the data file
    if split == "train":
        filename = "collected_train.json"
    elif split == "dev":
        filename = "collected_dev.json"
    elif split == "test":
        filename = "collected_test.json"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    full_path = os.path.join(data_path, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"MEQA data file not found: {full_path}")
    
    return MEQADataset(full_path, tokenizer, **kwargs)


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test loading
    dataset = load_meqa_dataset(
        data_path="/Users/yonganyu/Desktop/decore/MEQA/data",
        tokenizer=tokenizer,
        split="dev",
        num_samples=5
    )
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Test getting an item
    item = dataset[0]
    print(f"Example ID: {item['example_id']}")
    print(f"Question: {item['question']}")
    print(f"Answer: {item['answer']}")
    print(f"Explanation: {item['explanation']}")
    print(f"Prompted question shape: {item['prompted_question'].shape}")
