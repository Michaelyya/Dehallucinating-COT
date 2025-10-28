"""
Debug what the MEQA dataset is actually providing
"""

import sys
sys.path.append('/cluster/scratch/yongyu/decore')

from meqa_dataset import load_meqa_dataset
from transformers import AutoTokenizer
import torch

def debug_meqa_input():
    print("=== Debug MEQA Dataset Input ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_meqa_dataset(
        data_path="/cluster/scratch/yongyu/decore/MEQA/data",
        tokenizer=tokenizer,
        split="dev",
        num_samples=1
    )
    
    # Get first item
    item = dataset[0]
    
    print("=== MEQA Dataset Item ===")
    print(f"Example ID: {item['example_id']}")
    print(f"Question: {item['question']}")
    print(f"Answer: {item['answer']}")
    print()
    
    print("=== Input Fields ===")
    print(f"prompted_question type: {type(item['prompted_question'])}")
    print(f"prompted_question shape: {item['prompted_question'].shape}")
    print(f"verbalised_instruction: {item['verbalised_instruction']}")
    print(f"verbalised_question: {item['verbalised_question']}")
    print()
    
    # Decode the prompted_question to see what it contains
    if isinstance(item['prompted_question'], torch.Tensor):
        decoded = tokenizer.decode(item['prompted_question'][0], skip_special_tokens=False)
        print(f"Decoded prompted_question: {decoded[:500]}...")
    else:
        print(f"prompted_question (not tensor): {item['prompted_question']}")
    
    print()
    print("=== Testing with DeCoRe Framework ===")
    
    # Test with the actual MEQA input
    from src.configs import ModelConfigs, DecoderConfigs
    from src.factories import get_model
    
    # Create model configs
    model_configs = ModelConfigs(
        name="LLaMA3-8b-Instruct",
        model_type="instruct",
        configs=type('Config', (), {
            'model_name_or_path': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'max_seq_len': 4096,
            'max_new_tokens': 50
        })()
    )
    
    decoder_configs = DecoderConfigs(
        name="Baseline",
        method="Baseline",
        configs=type('Config', (), {})()
    )
    
    # Load model
    model = get_model(model_configs, decoder_configs)
    
    # Test generation with MEQA input
    result = model.generate(item)
    print(f"Generated result: '{result['decoded_text']}'")

if __name__ == "__main__":
    debug_meqa_input()
