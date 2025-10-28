"""
Debug the DeCoRe framework's _generate method
"""

import sys
sys.path.append('/cluster/scratch/yongyu/decore')

from transformers import AutoTokenizer
import torch
from src.configs import ModelConfigs, DecoderConfigs
from src.factories import get_model

def debug_decore_generation():
    print("=== Debug DeCoRe Generation ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configs
    model_configs = ModelConfigs(
        name="LLaMA3-8b-Instruct",
        model_type="instruct",
        configs=type('Config', (), {
            'model_name_or_path': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'max_seq_len': 4096,
            'max_new_tokens': 20  # Limit for debugging
        })()
    )
    
    decoder_configs = DecoderConfigs(
        name="Baseline",
        method="Baseline",
        configs=type('Config', (), {})()
    )
    
    # Load model
    model = get_model(model_configs, decoder_configs)
    
    # Test with simple input
    simple_input = {
        "prompted_question": ["What is 2+2?"],
        "verbalised_instruction": [""],
        "verbalised_icl_demo": [""],
        "verbalised_contexts": [""],
        "verbalised_question": ["What is 2+2?"],
        "verbalised_answer_prefix": [""],
    }
    
    print("Testing with simple input...")
    result = model.generate(simple_input)
    print(f"Result: '{result['decoded_text']}'")
    
    # Test with MEQA-style input
    meqa_input = {
        "prompted_question": ["Context: A drone was shot down. Question: What was destroyed? Answer:"],
        "verbalised_instruction": [""],
        "verbalised_icl_demo": [""],
        "verbalised_contexts": ["A drone was shot down."],
        "verbalised_question": ["What was destroyed?"],
        "verbalised_answer_prefix": [""],
    }
    
    print("\nTesting with MEQA-style input...")
    result2 = model.generate(meqa_input)
    print(f"Result: '{result2['decoded_text']}'")
    
    # Let's also test the _verbalise_input method directly
    print("\n=== Testing _verbalise_input ===")
    verbalised = model._verbalise_input("What is 2+2?", use_system_prompt=False, use_chat_template=False)
    print(f"Verbalised input shape: {verbalised.shape}")
    print(f"Verbalised input decoded: '{tokenizer.decode(verbalised[0])}'")

if __name__ == "__main__":
    debug_decore_generation()
