"""
MEQA Dataset for Multi-hop Event-centric Question Answering
"""

import json
import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import re


class MEQADataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_context_length: int = 4096,
        max_question_length: int = 512,
        max_explanation_length: int = 1024,
        use_chat_template: bool = True,
        num_samples: int = -1,
        # Controls for composed prompt/answer (mimicking reference prompting)
        compose_cot: bool = False,
        compose_exp: bool = True,
        compose_posthoc: bool = False,
        compose_graph: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.max_explanation_length = max_explanation_length
        self.use_chat_template = use_chat_template
        
        # Controls for composed prompt utilities
        self.compose_cot = compose_cot
        self.compose_exp = compose_exp
        self.compose_posthoc = compose_posthoc
        self.compose_graph = compose_graph
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
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
            # Format as a conversation for instruct models with explicit output schema
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
                        f"Context: {context}\n\n"
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
            
            # Return the text string, not tokenized - let DeCoRe framework handle tokenization
            prompted_question = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,  # Don't tokenize, return text
            )
        else:
            # Simple format for base models with explicit output schema
            prompted_question = (
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                "Respond in this exact format (do not add extra lines):\n"
                "Final answer: <few words only>\n"
                "Explanation:\n"
                "1. Evidence: \"<verbatim phrase from Context supporting the answer>\"\n"
                "2. Evidence: \"<another verbatim phrase from Context if helpful>\"\n"
                "3. Reasoning: <short 1-sentence link from evidence to the answer>\n\n"
                "Answer:"
            )
        
        composed_prompt = compose_prompt(
            {
                "context": context,
                "question": question,
                "answer": answer,
                "explanation": explanation,
                # Optional fields for graph-related prompts if present in data
                **({"events": item.get("events")} if "events" in item else {}),
                **({"relation": item.get("relation")} if "relation" in item else {}),
                **({"entity_graph": item.get("entity_graph")} if "entity_graph" in item else {}),
            },
            cot=self.compose_cot,
            exp=self.compose_exp,
            posthoc=self.compose_posthoc,
            graph=self.compose_graph,
        )
        
        # Create reference explanation (for evaluation)
        reference_explanation = " ".join(explanation)
        
        # A composed reference-style answer string (for parity with reference utilities)
        composed_reference_answer = compose_answer(
            {
                "answer": answer,
                "explanation": explanation,
                # Optional field if reasoning_chain is provided in data
                **({"reasoning_chain": item.get("reasoning_chain")} if "reasoning_chain" in item else {}),
            },
            cot=self.compose_cot,
            exp=self.compose_exp,
            posthoc=self.compose_posthoc,
        )
        
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
            # Additional fields to support reference-style prompting and downstream analysis
            "composed_prompt": composed_prompt,
            "composed_reference_answer": composed_reference_answer,
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


def generate_entity_graph_prompt(d: Dict) -> str:
    document = '\n'.join(['Document:', d.get('context', '')])
    events = d.get('events', []) or []
    entity_list: List[str] = []
    for event in events:
        for argument in event.get('arguments', []) or []:
            entity_list.append(argument.get('text', ''))
    entity_list = list(set([e for e in entity_list if e]))
    entities = '\n'.join(entity_list)
    entities = '\n'.join(['Entities:', entities])
    graph_prompt = 'Given the document, please output shorten relations only between above entities in (entity, relation, entity) format:'
    if 'entity_graph' in d and d['entity_graph']:
        graph = '\n'.join([graph_prompt, d['entity_graph']])
    else:
        graph = graph_prompt
    prompt = '\n\n'.join([document, entities, graph])
    return prompt


def compose_prompt(d: Dict, cot: bool = False, exp: bool = False, posthoc: bool = False, graph: str = '') -> str:
    message = ''

    if not cot:
        if not exp:
            # plain qa
            question = '\n'.join(['Please answer the question:', d.get('question', ''), 'Answer:'])
        else:
            if not posthoc:
                # cot
                question = '\n'.join(['Please decompose and answer the question:', d.get('question', ''), 'Answer:'])
            else:
                # posthoc
                question = '\n'.join(['Please answer the question and then explain the answer:', d.get('question', ''), 'Answer:'])
    else:
        # free-form cot
        question = '\n'.join(['Please answer the question step-by-step:', d.get('question', ''), 'Answer:'])
    document = '\n'.join(['Document:', d.get('context', '')])

    if 'golden-event' == graph:
        events = d.get('events', []) or []
        triples: List[str] = []
        # connect all arguments and events
        for event in events:
            trigger = (event.get('trigger') or {}).get('text', '')
            for argument in event.get('arguments', []) or []:
                triple = '({trigger}, {rel}, {arg})'.format(
                                trigger=trigger,
                                rel=argument.get('role', ''),
                                arg=argument.get('text', '')
                            )
                triples.append(triple)
        # for strategy 1, connect events by event relations
        if 'relation' in d and len(events) >= 2:
            trigger1 = (events[0].get('trigger') or {}).get('text', '')
            trigger2 = (events[1].get('trigger') or {}).get('text', '')
            triple = '({trigger1}, {rel}, {trigger2})'.format(
                            trigger1=trigger1,
                            rel=d.get('relation', ''),
                            trigger2=trigger2
                        )
            triples.append(triple)
        triples_joined = '\n'.join(triples)
        triples_block = '\n'.join(['Graph:', triples_joined])
        if "" == message:
            message = '\n\n'.join([document, triples_block, question])
        else:
            message = '\n\n'.join([message, document, triples_block, question])
    elif 'golden-trigger-arguments' == graph:
        events = d.get('events', []) or []
        triples: List[str] = []
        # connect all arguments and events
        for event in events:
            trigger = (event.get('trigger') or {}).get('text', '')
            for argument in event.get('arguments', []) or []:
                triple = '({trigger}, {arg})'.format(
                                trigger=trigger,
                                arg=argument.get('text', '')
                            )
                triples.append(triple)
        # for strategy 1, connect events by event relations
        if 'relation' in d and len(events) >= 2:
            trigger1 = (events[0].get('trigger') or {}).get('text', '')
            trigger2 = (events[1].get('trigger') or {}).get('text', '')
            triple = '({trigger1}, {trigger2})'.format(
                            trigger1=trigger1,
                            trigger2=trigger2
                        )
            triples.append(triple)
        triples_joined = '\n'.join(triples)
        triples_block = '\n'.join(['Graph:', triples_joined])
        if "" == message:
            message = '\n\n'.join([document, triples_block, question])
        else:
            message = '\n\n'.join([message, document, triples_block, question])
    # intentionally omit 'predict-entity-graph' for a clean, dependency-free codepath
    elif 'golden-entity' == graph:
        events = d.get('events', []) or []
        entity_list: List[str] = []
        for event in events:
            for argument in event.get('arguments', []) or []:
                entity_list.append(argument.get('text', ''))
        entity_list = list(set([e for e in entity_list if e]))
        entities = ', '.join(entity_list)
        entities_block = '\n'.join(['Entities:', entities])
        if "" == message:
            message = '\n\n'.join([document, entities_block, question])
        else:
            message = '\n\n'.join([message, document, entities_block, question])
    elif 'golden-trigger' == graph:
        events = d.get('events', []) or []
        trigger_list: List[str] = []
        for event in events:
            trigger = (event.get('trigger') or {}).get('text', '')
            if trigger:
                trigger_list.append(trigger)
        triggers = ', '.join(trigger_list)
        triggers_block = '\n'.join(['Triggers:', triggers])
        if "" == message:
            message = '\n\n'.join([document, triggers_block, question])
        else:
            message = '\n\n'.join([message, document, triggers_block, question])
    elif '' == graph:
        if "" == message:
            message = '\n\n'.join([document, question])
        else:
            message = '\n\n'.join([message, document, question])

    return message


def compose_example(d: Dict, cot: bool = False, exp: bool = False, posthoc: bool = False, graph: str = '') -> str:
    message = compose_prompt(d, cot=cot, exp=exp, posthoc=posthoc, graph=graph)
    answer = compose_answer(d, cot=cot, exp=exp, posthoc=posthoc)

    message = '\n'.join([message, answer])
    return message


def compose_answer(d: Dict, cot: bool = False, exp: bool = False, posthoc: bool = False) -> str:
    raw_answer = d.get('answer', '')
    # Normalize answer to string (handles list or other types)
    if isinstance(raw_answer, list):
        answer_text = ', '.join([str(a) for a in raw_answer if a is not None])
    else:
        answer_text = str(raw_answer)
    answer = answer_text
    explanation = ''
    for idx, explain in enumerate(d.get('explanation', []) or []):
        explain = re.sub('@\d+', '', explain)
        explanation += str(idx+1) + '. ' + explain + '\n'
    # add the answer if the explanation doesn't have one
    if explanation.strip().endswith('?'):
        explanation += ' ' + answer_text

    if not cot:
        if not exp:
            # plain qa A
            answer = answer_text
        else:
            if not posthoc:
                # cot E+A
                answer = explanation
                answer += 'So, the answer is: ' + answer_text + '.'
            else:
                # post-hoc A+E
                explanation = '\n'.join(['Explanation:', explanation])
                answer = '\n'.join([answer_text, explanation])
    else:
        # free-form cot FE+A
        answer = d.get('reasoning_chain', d.get('answer', ''))
    return answer


def extract_answer(prediction: str, cot: bool = False, exp: bool = False, posthoc: bool = False):
    prediction_ori = prediction
    if not cot:
        # extract from common answer patterns
        extract = (
            re.search(r'(?:final\s+)?answer\s*:\s*([^\n]+)', prediction, re.IGNORECASE)
            or re.search(r'the\s+answer\s+is\s*[:]?\s*([^\n]+)', prediction, re.IGNORECASE)
        )
        if extract is not None:
            prediction = extract.group(1)
            prediction = re.sub(r'^\W+|\W+$', '', prediction)

        if exp:
            if not posthoc:
                # cot E+A
                if prediction == prediction_ori:
                    explanation = prediction.split('\n')
                    # extract from explanation pattern
                    extract = re.search('(.*)?([^?]+)$', prediction, re.IGNORECASE)
                    if extract is not None:
                        prediction = extract.group(2)
                        prediction = re.sub(r'^\W+|\W+$', '', prediction)
                else:
                    explanation = prediction_ori.split('\n')[:-1]
                return prediction, explanation
            else:
                # Post-hoc A+E
                # extract answer sentence
                extract = re.search('((.|\n)*)Explanation:((.|\n)*)$', prediction, re.IGNORECASE)
                explanation: List[str] = []
                if extract is not None:
                    prediction = extract.group(1)
                    prediction = re.sub(r'^\W+|\W+$', '', prediction)
                    explanation = extract.group(3).split('\n')
                    return prediction, explanation
    else:
        # free-form cot FE+A
        # extract from pattern variants
        extract = (
            re.search(r'(?:final\s+)?answer\s*:\s*([^\n]+)', prediction, re.IGNORECASE)
            or re.search(r'the\s+answer\s+is\s*[:]?\s*([^\n]+)', prediction, re.IGNORECASE)
        )
        if extract is not None:
            prediction = extract.group(1)
            prediction = re.sub(r'^\W+|\W+$', '', prediction)

    prediction = re.sub(r'^\W+|\W+$', '', prediction)
    return prediction



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
