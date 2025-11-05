"""
Modified script for detecting reasoning heads instead of retrieval heads.
This script analyzes attention patterns during reasoning tasks to identify 
which attention heads are most active during the reasoning process.
"""

import os 
import glob
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, OPTForCausalLM
import sys
sys.path.append("./faiss_attn/")
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
#from source.modeling_phi3 import Phi3ForCausalLM
import numpy as np
import argparse
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch


def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


class LLMReasoningHeadDetector:
    """
    This class is used to detect reasoning heads in LLMs by analyzing attention patterns
    during reasoning tasks with query, ground truth answer, and context.
    """
    def __init__(self,
                reasoning_data_path="./collected_train.json",
                results_version = 1,
                context_lengths_min = 1000,
                context_lengths_max = 50000,
                context_lengths_num_intervals = 20,
                context_lengths = None,
                model_provider = "LLaMA",
                model_name='',
                model_name_suffix=None,
                save_results = True,
                save_contexts = True,
                final_context_length_buffer = 200,
                print_ongoing_status = True):
        """        
        :param reasoning_inputs_path: Path to JSONL file containing reasoning inputs
                                      Each line: {"id": str, "query": str, "context": str, "task": str}
        :param reasoning_groundtruth_path: Path to JSONL with ground truth answers
                                      Each line: {"id": str, "ground_truth": str}
        :param model_name: The name/path of the model
        :param save_results: Whether to save results
        :param print_ongoing_status: Whether to print ongoing status
        """
        
        # Load reasoning inputs
        self.reasoning_samples = []
        self.id_to_groundtruth = {}

        if os.path.exists(reasoning_data_path):
            with open(reasoning_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f) 
        
            for item in data:
                if all(k in item for k in ['example_id', 'context', 'question', 'answer']):
                 
                    sample = {
                        'id': item['example_id'],
                        'query': item['question'],  
                        'context': item['context']
                    }
                    self.reasoning_samples.append(sample)
            
                   
                    self.id_to_groundtruth[item['example_id']] = item['answer']
        
        # Sample 5 samples for testing
        import random
        random.seed(42)
        self.reasoning_samples = random.sample(self.reasoning_samples, 5)

        print(f"Loaded {len(self.reasoning_samples)} reasoning samples from {reasoning_data_path}")
        print(f"Ground truth answers loaded: {len(self.id_to_groundtruth)}")
        
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.head_counter = defaultdict(list)  # Store reasoning head activation scores
        
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: 
            self.model_version = model_name
        if(model_name_suffix is not None): 
            self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Context length parameters must be specified")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths
        
        self.model_name = model_name
        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"Loading model from {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads
        print(f"Model has {self.layer_num} layers and {self.head_num} attention heads")

        if "opt-125m" in self.model_version.lower() or "opt" in model_name.lower():
            self.model_to_test = OPTForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map='auto'
            ).eval()
        elif "Qwen" in self.model_version:
            self.model_to_test = Qwen2ForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map='auto', # use_flash_attention_2="flash_attention_2",
            ).eval()
        elif "Mixtral" in self.model_version:
            self.model_to_test = MixtralForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map='auto', # use_flash_attention_2="flash_attention_2",
                trust_remote_code=True
            ).eval()
        elif "Mistral" in self.model_version:
            self.model_to_test = MistralForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map='auto', # use_flash_attention_2="flash_attention_2",
                trust_remote_code=True
            ).eval()
        elif "Phi3" in self.model_version:
            self.model_to_test = Phi3ForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map='auto', # use_flash_attention_2="flash_attention_2",
                trust_remote_code=True
            ).eval()
        else:
      
            self.model_to_test = LlamaForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map='auto',
                attn_implementation="eager" 
            ).eval()
            
        if 'llama-2-7b-80k' in self.model_version:
            scaling_factor = 10
            reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
            
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1
        else:
            self.multi_gpus = True
            
        self.model_to_test_description = model_name

    def reasoning_calculate(self, attention_matrix, reasoning_score, inp, step_token, prompt_len, gen_len, topk=5):
        """
        Calculate reasoning head scores by analyzing attention to ground truth tokens
        during the generation process.
        Reasoning head: checks if attention focuses on previously generated tokens (chain-of-thought)
        
        :param attention_matrix: Attention weights from the model
        :param reasoning_score: Current reasoning scores for each head
        :param inp: Current input token
        :param step_token: Current step token string
        :param prompt_len: Length of the prompt (input) tokens
        :param gen_len: Number of tokens generated so far (before current token)
        :param topk: Number of top attention positions to consider
        """
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                # Get top-k attention positions for this head
                values, idx = attention_matrix[layer_idx][0][head_idx][-1].topk(topk)
                
                # Check if attention focuses on reasoning-relevant tokens
                reasoning_attention_score = 0
                for v, i in zip(values, idx):
                    # Attend to previously generated tokens (chain-of-thought)
                    if i >= prompt_len and i < prompt_len + gen_len:
                        reasoning_attention_score += v.item()

                    # Also consider attention to the query span
                    if hasattr(self, 'query_start') and hasattr(self, 'query_end'):
                        if self.query_start <= i < self.query_end:
                            reasoning_attention_score += v.item() * 0.5  
                
                # Update reasoning score for this head
                if reasoning_attention_score > 0:
                    reasoning_score[layer_idx][head_idx][0] += reasoning_attention_score
                    reasoning_score[layer_idx][head_idx][1] += step_token

    def reasoning_head_accumulate(self, reasoning_score):
        """
        Accumulate reasoning scores across multiple samples to identify consistent reasoning heads.
      
        """
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                score = reasoning_score[layer_idx][head_idx][0]
                if score > 0: 
                    self.head_counter[f"{layer_idx}-{head_idx}"].append(score)

    def decode_with_reasoning_tracking(self, q_outputs, inp, decode_len):
        """
        Decode tokens while tracking reasoning head activation patterns.
        Follow same way of Retrieval_head paper: （past_key_values + output_attentions=True）
        :param q_outputs: Model outputs from prompt encoding
        :param inp: Initial input token
        :param decode_len: Maximum number of tokens to decode
        :param ground_truth_ids: Ground truth answer token IDs for comparison
        :return: Generated output tokens and reasoning scores
        """
        output = []
        reasoning_score = [[[0, ''] for _ in range(self.head_num)] for _ in range(self.layer_num)]
        past_kv = q_outputs.past_key_values
        prompt_len = len(self.prompt_ids)
        
        for step_i in range(decode_len):
            inp = inp.view(1, 1)
            outputs = self.model_to_test(
                input_ids=inp,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            
            # gen_len is number of tokens already generated before current token
            self.reasoning_calculate(outputs.attentions, reasoning_score, inp, step_token, prompt_len, gen_len=len(output)-1)
            
            if step_token == '<0x0A>' or inp.item() == 144 or inp.item() == self.enc.eos_token_id: 
                break
                
        return output, reasoning_score

    def find_query_idx(self, query):
        """
        Find the token span of the query in the prompt for attention analysis.

        """
        query_ids = self.enc(query, add_special_tokens=False)["input_ids"]
        span_len = len(query_ids)
        
        for i in range(len(self.prompt_ids)):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(query_ids)))) / len(set(query_ids))
            if overlap > 0.8:  # Allow some flexibility
                return i, i + span_len
        return -1, -1

    def evaluate_reasoning_sample(self, sample, context_length):
        """
        Evaluate a single reasoning sample and track head activations.
        
        :param sample: Dictionary with 'query', 'ground_truth', and 'context'
        :param context_length: Target context length
        """
        sample_id = sample['id']
        query = sample['query']
        context = sample['context']
        ground_truth = self.id_to_groundtruth.get(sample_id, "")
        
        # Try diff context length
        context_tokens = self.enc(context, add_special_tokens=False)['input_ids']
        if len(context_tokens) > context_length - self.final_context_length_buffer:
            context_tokens = context_tokens[:context_length - self.final_context_length_buffer]
            context = self.enc.decode(context_tokens)
        
        # Construct prompt based on model type
        if self.model_version in ["Mistral-7B-Instruct-v0.2", "Qwen1.5-14B-Chat"]:
            prompt = [
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            input_ids = self.enc.apply_chat_template(
                conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            )
        else:
            input_context = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            input_ids = self.enc(input_context, return_tensors="pt")['input_ids']
        
        # Prepare for evaluation
        test_start_time = time.time()
        self.prompt_ids = input_ids[0, :]
        
        if not self.multi_gpus:
            input_ids = input_ids.to(self.model_to_test.device)
        
        # Find query position in prompt
        self.query_start, self.query_end = self.find_query_idx(query)
        
        # Generate and track reasoning
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True, output_attentions=True)
            output, reasoning_score = self.decode_with_reasoning_tracking(q_outputs, input_ids[:,-1], 100)
            response = self.enc.decode(output, skip_special_tokens=True).strip()
        
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        
    
        score = scorer.score(ground_truth, response)['rouge1'].recall * 100 if ground_truth else 0.0
        
    
        if score > 50:
            self.reasoning_head_accumulate(reasoning_score)
            
            # Print top reasoning heads
            head_scores = [(head_id, np.mean(scores)) for head_id, scores in self.head_counter.items() if scores]
            head_scores = sorted(head_scores, key=lambda x: x[1], reverse=True)
            print(f"\nTop 20 reasoning heads: {[h[0] for h in head_scores[:20]]}")
            print(f"Score: {score:.2f}, Response length: {len(output)} tokens")
        
        # Store results
        results = {
            'model': self.model_to_test_description,
            'context_length': int(context_length),
            'id': sample_id,
            'query': query,
            'ground_truth': ground_truth,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }
        
        self.testing_results.append(results)
        
        if self.print_ongoing_status:
            print(f"\n-- Reasoning Test Summary --")
            print(f"Duration: {test_elapsed_time:.1f}s")
            print(f"Context: {context_length} tokens")
            print(f"Score: {score:.2f}")
            print(f"Response: {response[:100]}...")
        
        return results

    def run_test(self):
        """
        Run reasoning head detection on all samples across different context lengths.
        """
        for context_length in self.context_lengths:
            print(f"\n{'='*60}")
            print(f"Testing with context length: {context_length}")
            print(f"{'='* 60}")
            
            for idx, sample in enumerate(self.reasoning_samples):
                print(f"\nProcessing sample {idx+1}/{len(self.reasoning_samples)}")
                self.evaluate_reasoning_sample(sample, context_length)

    def save_reasoning_heads(self):
        """
        Save detected reasoning heads to file.
        """
        output_dir = "reasoning_head_scores"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/{self.model_version}_reasoning_heads.json"
        
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing_scores = json.load(f)
            # Merge with current scores
            for k, v in existing_scores.items():
                self.head_counter[k].extend(v)
        
        # Save updated scores
        with open(output_path, 'w') as f:
            json.dump(self.head_counter, f, indent=2)
        
        summary = {}
        for head_id, scores in self.head_counter.items():
            if scores:
                summary[head_id] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores),
                    'max': float(np.max(scores))
                }
        
        summary_path = f"{output_dir}/{self.model_version}_reasoning_heads_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nReasoning head scores saved to {output_path}")
        print(f"Summary statistics saved to {summary_path}")

    def print_start_test_summary(self):
        print("\n" + "="*60)
        print("Starting Reasoning Head Detection...")
        print(f"- Model: {self.model_name}")
        print(f"- Number of reasoning samples: {len(self.reasoning_samples)}")
        print(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print("="*60 + "\n")

    def start_test(self):
        """
        Main entry point to start reasoning head detection.
        """
        if self.print_ongoing_status:
            self.print_start_test_summary()
        
        self.run_test()
        self.save_reasoning_heads()
        
        # Print final top reasoning heads
        head_scores = [(head_id, np.mean(scores)) for head_id, scores in self.head_counter.items() if scores]
        head_scores = sorted(head_scores, key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*60)
        print("Top 30 Reasoning Heads (Layer-Head: Mean Score):")
        print("="*60)
        for i, (head_id, score) in enumerate(head_scores[:30], 1):
            layer, head = head_id.split('-')
            print(f"{i:2d}. Layer {layer:2s}, Head {head:2s}: {score:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect reasoning heads in LLMs')
    parser.add_argument('--reasoning_data', type=str, default='./collected_train.json',
                       help='Path to reasoning data JSON file')
    parser.add_argument('-s', '--s_len', type=int, default=1000,
                       help='Minimum context length')
    parser.add_argument('-e', '--e_len', type=int, default=50000,
                       help='Maximum context length')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model (e.g., facebook/opt-125m)')
    parser.add_argument('--model_name_suffix', type=str, default=None,
                       help='Suffix for model name in results')
    parser.add_argument('--model_provider', type=str, default="LLaMA",
                       help='Model provider (LLaMA, Mistral, OPT, etc.)')
    
    args = parser.parse_args()
    
    

    detector = LLMReasoningHeadDetector(
        reasoning_data_path=args.reasoning_data,
        model_name=args.model_path,
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_results=True,
        save_contexts=True,
        context_lengths_min=args.s_len,
        context_lengths_max=args.e_len,
    )
    
    detector.start_test()