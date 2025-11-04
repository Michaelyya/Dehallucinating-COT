"""
MuSiQue Dataset for Multi-hop Question Answering
"""

import json
import os
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import re


class MuSiQueDataset(Dataset):
    
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
        
        # Load JSONL data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"MuSiQue data file not found: {data_path}")
        
        self.data: List[Dict] = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        
        if num_samples > 0:
            self.data = self.data[:num_samples]
        
        print(f"Loaded {len(self.data)} MuSiQue examples from {data_path}")
    
    def _flatten_context(self, paragraphs: List[Dict]) -> str:
        parts: List[str] = []
        for para in paragraphs:
            title = para.get('title', '')
            text = para.get('paragraph_text', '')
            parts.append(f"{title}: {text}")
        return "\n\n".join(parts)
    
    def _format_supporting_explanation(self, paragraphs: List[Dict]) -> str:
        lines: List[str] = []
        for para in paragraphs:
            if para.get('is_supporting', False):
                title = para.get('title', '')
                text = para.get('paragraph_text', '')
                lines.append(f"{title}: {text}")
        if not lines:
            return ""
        return "\n".join([f"{i+1}. {t}" for i, t in enumerate(lines)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        example_id = item.get('id', str(idx))
        question = item.get('question', '')
        answer = item.get('answer', '')
        paragraphs = item.get('paragraphs', [])
        
        context_text = self._flatten_context(paragraphs)
        explanation = self._format_supporting_explanation(paragraphs)
        reference_explanation = explanation
        
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
                        "Respond in this exact format (do not add extra lines):\n"
                        "Final answer: <few words only>\n"
                        "Explanation:\n"
                        "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
                        "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
                        "3. Reasoning: <short 1-sentence link from evidence to the answer>\n\n"
                        "Answer:"
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
                "Respond in this exact format (do not add extra lines):\n"
                "Final answer: <few words only>\n"
                "Explanation:\n"
                "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
                "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
                "3. Reasoning: <short 1-sentence link from evidence to the answer>\n\n"
                "Answer:"
            )
        
        return {
            "idx": idx,
            "example_id": example_id,
            "prompted_question": prompted_question,
            "context": context_text,
            "question": question,
            "answer": answer,
            "reference_explanation": reference_explanation,
            # Compatibility fields
            "verbalised_instruction": [""],
            "verbalised_icl_demo": [""],
            "verbalised_contexts": [context_text],
            "verbalised_question": [question],
            "verbalised_answer_prefix": [""],
        }
    
    def collate_fn(self, batch):
        first_item = batch[0]
        collated: Dict[str, List] = {}
        for key in first_item.keys():
            collated[key] = [item[key] for item in batch]
        return collated


def extract_answer(prediction: str) -> str:
    prediction_ori = prediction
    extract = (
        re.search(r'(?:final\s+)?answer\s*:\s*([^\n]+)', prediction, re.IGNORECASE)
        or re.search(r'the\s+answer\s+is\s*[:]?\s*([^\n]+)', prediction, re.IGNORECASE)
    )
    if extract is not None:
        prediction = extract.group(1)
        prediction = re.sub(r'^\W+|\W+$', '', prediction)
    prediction = re.sub(r'^\W+|\W+$', '', prediction)
    if not prediction:
        patterns = [
            r'(?:final\s+)?answer:\s*(.+?)(?:\n|$)',
            r'answer:\s*(.+?)(?:\n|$)',
            r'the\s+answer\s+is:\s*(.+?)(?:\n|$)'
        ]
        for p in patterns:
            m = re.search(p, prediction_ori, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
    return prediction


def load_musique_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "dev",
    **kwargs
) -> MuSiQueDataset:
    """
    Load MuSiQue dataset from a JSONL file path
    """
    # Expect a file path; allow directory + default name
    if os.path.isdir(data_path):
        if split == "dev":
            filename = "musique_ans_v1.0_dev.jsonl"
        elif split == "train":
            filename = "musique_ans_v1.0_train.jsonl"
        elif split == "test":
            filename = "musique_ans_v1.0_test.jsonl"
        else:
            raise ValueError(f"Unknown split: {split}")
        full_path = os.path.join(data_path, filename)
    else:
        full_path = data_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"MuSiQue data file not found: {full_path}")
    return MuSiQueDataset(full_path, tokenizer, **kwargs)


