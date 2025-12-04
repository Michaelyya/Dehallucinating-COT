"""
HotpotQA Dataset for Multi-hop Question Answering
"""

import json
import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import re


class HotpotQADataset(Dataset):
    
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
        
        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        print(f"Loaded {len(self.data)} HotpotQA examples from {data_path}")
    
    def _flatten_context(self, context: List[List]) -> str:
        """Convert HotpotQA context format [[title, sentences], ...] to a single string"""
        paragraphs = []
        for para in context:
            title = para[0]
            sentences = para[1]
            para_text = f"{title}: " + " ".join(sentences)
            paragraphs.append(para_text)
        return "\n\n".join(paragraphs)
    
    def _format_supporting_facts(self, supporting_facts: List[List], context: List[List]) -> str:
        """Format supporting facts as explanation text"""
        if not supporting_facts:
            return ""
        
        fact_texts = []
        for fact in supporting_facts:
            title = fact[0]
            sent_id = fact[1]
            
            # Find the sentence in context
            for para in context:
                if para[0] == title and sent_id < len(para[1]):
                    sentence = para[1][sent_id]
                    fact_texts.append(f"{title}: {sentence}")
        
        return "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(fact_texts)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract components
        example_id = item["_id"]
        question = item["question"]
        answer = item.get("answer", "")
        supporting_facts = item.get("supporting_facts", [])
        context = item["context"]
        
        # Flatten context to single string
        context_text = self._flatten_context(context)
        
        # Format supporting facts as explanation
        explanation = self._format_supporting_facts(supporting_facts, context)
        reference_explanation = explanation
        
        # Create prompt for CoT reasoning
        if self.use_chat_template:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant. Follow the output schema exactly. "
                        "Use only evidence from the provided Context (no external knowledge). "
                        "Keep the final answer to a few words. In Explanation, keep each line concise."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context: {context_text}\n\n"
                        f"Question: {question}\n\n"
                        "First, provide your answer in a few words only (no prefix, just the answer).\n"
                        "Then, provide an explanation in this format:\n"
                        "Explanation:\n"
                        "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
                        "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
                        "3. Reasoning: <short 1-sentence link from evidence to the answer>"
                    )
                }
            ]
            
            prompted_question = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompted_question = (
                f"Context: {context_text}\n\n"
                f"Question: {question}\n\n"
                "First, provide your answer in a few words only (no prefix, just the answer).\n"
                "Then, provide an explanation in this format:\n"
                "Explanation:\n"
                "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
                "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
                "3. Reasoning: <short 1-sentence link from evidence to the answer>"
            )
        
        return {
            "idx": idx,
            "example_id": example_id,
            "prompted_question": prompted_question,
            "context": context_text,
            "question": question,
            "answer": answer,
            "supporting_facts": supporting_facts,
            "reference_explanation": reference_explanation,
            # Additional fields for compatibility with existing framework
            "verbalised_instruction": [""],
            "verbalised_icl_demo": [""],
            "verbalised_contexts": [context_text],
            "verbalised_question": [question],
            "verbalised_answer_prefix": [""],
        }
    
    def collate_fn(self, batch):
        """Custom collate function for HotpotQA data"""
        first_item = batch[0]
        
        collated = {}
        for key in first_item.keys():
            collated[key] = [item[key] for item in batch]
        
        return collated


def extract_answer(prediction: str) -> str:
    """Extract answer from format: <answer>\nExplanation:..."""
    prediction = prediction.strip()
    
    # First, try to extract everything before "Explanation:"
    explanation_match = re.search(r'Explanation\s*:', prediction, re.IGNORECASE)
    if explanation_match:
        answer = prediction[:explanation_match.start()].strip()
        if answer:
            return answer
    
    # Fallback: extract from common patterns (for backward compatibility)
    patterns = [
        r'(?:final\s+)?answer\s*:\s*([^\n]+)',
        r'the\s+answer\s+is\s*[:]?\s*([^\n]+)',
        r'^([^\n]+?)(?:\n|$)'  # First line if no pattern matches
    ]
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Remove common prefixes
            answer = re.sub(r'^(final\s+)?answer\s*:\s*', '', answer, flags=re.IGNORECASE)
            if answer and answer.lower() not in ['none', 'n/a', 'unknown']:
                return answer
    
    # Last resort: return first line
    first_line = prediction.split('\n')[0].strip()
    return first_line if first_line else ""


def load_hotpotqa_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "dev",
    **kwargs
) -> HotpotQADataset:
    """
    Load HotpotQA dataset from the specified path
    
    Args:
        data_path: Path to the HotpotQA data directory or file
        tokenizer: Tokenizer to use
        split: Dataset split ('train', 'dev', 'test')
        **kwargs: Additional arguments for HotpotQADataset
    
    Returns:
        HotpotQADataset instance
    """
    # Determine filename based on split
    if split == "train":
        filename = "hotpot_train_v1.1.json"
    elif split == "dev":
        filename = "hotpot_dev_distractor_v1.json"
    elif split == "test":
        filename = "hotpot_test_fullwiki_v1.json"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Check if data_path is a directory or file
    if os.path.exists(data_path):
        if os.path.isdir(data_path):
            # If directory, append filename
            full_path = os.path.join(data_path, filename)
        elif os.path.isfile(data_path):
            # If file, use directly
            full_path = data_path
        else:
            raise ValueError(f"Path exists but is neither a file nor directory: {data_path}")
    else:
        # Path doesn't exist, assume it's meant to be a directory and construct full path
        full_path = os.path.join(data_path, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"HotpotQA data file not found: {full_path}\n"
            f"Please ensure the HotpotQA data file exists at this path."
        )
    
    return HotpotQADataset(full_path, tokenizer, **kwargs)

