"""
Test script for HotpotQA dataset
"""

import os
import json
import argparse
import yaml
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import sys
sys.path.append('/cluster/scratch/yongyu/decore')

from src.models.base_model import BaseModel
from src.factories import get_model
from src.configs import ModelConfigs, DecoderConfigs, DataLoaderConfigs
from transformers import AutoTokenizer
from dataclasses import dataclass
from hotpotqa_dataset import load_hotpotqa_dataset, extract_answer
from hotpotqa_metrics import HotpotQAMetrics
from types import SimpleNamespace


@dataclass
class ModelConfig:
    model_name_or_path: str
    max_seq_len: int = 4096
    max_new_tokens: int = 100


@dataclass
class DecoderConfig:
    retrieval_heads_dir: str = "./retrieval_heads/"
    num_retrieval_heads: int = 10
    post_softmax: bool = True
    alpha_cap: float = None
    scale_alpha: bool = False
    amateur_model_name_or_path: str = None


def _dict_to_namespace_with_get(d: Dict[str, Any]) -> Any:
    """Wrap a dict so it supports both attribute access and dict-like get()."""
    ns = SimpleNamespace(**d)
    def _get(key: str, default=None):
        return getattr(ns, key, default)
    setattr(ns, 'get', _get)
    return ns


class HotpotQATester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["configs"]["model_name_or_path"]
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        model_config = config["model"]
        model_configs_dict = model_config["configs"]
        self.model_configs = ModelConfigs(
            name=model_config["name"],
            model_type=model_config["model_type"],
            configs=ModelConfig(
                model_name_or_path=model_configs_dict["model_name_or_path"],
                max_seq_len=model_configs_dict.get("max_seq_len", 4096),
                max_new_tokens=model_configs_dict.get("max_new_tokens", 100)
            )
        )
        
        decoder_config = config["decoder"]
        decoder_configs_dict = decoder_config.get("configs", {})
        
        # For DeCoRe models, pass configs as namespace; for baseline, use DecoderConfig
        if "decore" in decoder_config["method"].lower():
            self.decoder_configs = DecoderConfigs(
                name=decoder_config["name"],
                method=decoder_config["method"],
                configs=_dict_to_namespace_with_get(decoder_configs_dict)
            )
        else:
            self.decoder_configs = DecoderConfigs(
                name=decoder_config["name"],
                method=decoder_config["method"],
                configs=DecoderConfig(
                    retrieval_heads_dir=decoder_configs_dict.get("retrieval_heads_dir", "./retrieval_heads/"),
                    num_retrieval_heads=decoder_configs_dict.get("num_retrieval_heads", 10),
                    post_softmax=decoder_configs_dict.get("post_softmax", True),
                    alpha_cap=decoder_configs_dict.get("alpha_cap", None),
                    scale_alpha=decoder_configs_dict.get("scale_alpha", False),
                    amateur_model_name_or_path=decoder_configs_dict.get("amateur_model_name_or_path", None)
                )
            )
        
        self.model = get_model(self.model_configs, self.decoder_configs)
        
        # Initialize dataset
        hotpotqa_data_path = os.path.join("/cluster/scratch/yongyu/decore", config["data"]["data_dir"])
        self.dataset = load_hotpotqa_dataset(
            data_path=hotpotqa_data_path,
            tokenizer=self.tokenizer,
            split=config["data"]["split"],
            num_samples=config["data"]["num_samples"],
            max_context_length=config["data"]["max_context_length"],
            max_question_length=config["data"]["max_question_length"],
            max_explanation_length=config["data"]["max_explanation_length"],
            use_chat_template=config["data"]["use_chat_template"],
        )
        
        # Initialize data loader
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config["data_loader"]["batch_size"],
            shuffle=False,
            num_workers=config["data_loader"]["num_workers"],
            drop_last=config["data_loader"]["drop_last"],
            pin_memory=config["data_loader"]["pin_memory"],
            collate_fn=self.dataset.collate_fn,
        )
        
        self.metrics = HotpotQAMetrics()
        self.output_dir = config["evaluation"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize WandB if not in debug mode
        if not config.get("debug", True):
            wandb.init(
                project=config["wandb_project"],
                entity=config["wandb_entity"],
                config=config,
                name=f"HotpotQA_{config['decoder']['name']}_{config['data']['split']}"
            )
    
    def test(self) -> Dict[str, Any]:
        print(f"Starting HotpotQA testing with {len(self.dataset)} examples")
        print(f"Model: {self.model_configs.name}")
        print(f"Decoder: {self.decoder_configs.name}")
        print(f"Device: {self.device}")
        
        all_predictions = []
        
        # Run inference
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Testing")):
                # Generate predictions
                prediction = self.model.generate(batch)
                
                # Process batch results
                batch_size = len(batch["example_id"])
                for i in range(batch_size):
                    # Get full text generation
                    full_output = prediction["decoded_text"][i] if isinstance(prediction["decoded_text"], list) else prediction["decoded_text"]
                    
                    # Extract concise answer
                    short_answer = extract_answer(full_output)
                    
                    # Fallback to metrics' extractor if empty
                    if not short_answer:
                        short_answer = self.metrics.extract_answer_from_response(full_output)
                    
                    # For supporting facts, we'd need to extract from the explanation
                    # For now, use empty list (can be extended to parse from explanation)
                    predicted_supporting_facts = []
                    
                    pred_dict = {
                        "example_id": batch["example_id"][i],
                        "question": batch["question"][i],
                        "context": batch["context"][i],
                        "predicted_answer": short_answer,
                        "answer": batch["answer"][i],
                        "supporting_facts": batch["supporting_facts"][i],
                        "predicted_supporting_facts": predicted_supporting_facts,
                        "predicted_explanation": full_output,
                    }
                    
                    # Add attention information if available
                    if "alphas" in prediction and prediction["alphas"] is not None:
                        pred_dict["alphas"] = prediction["alphas"][i] if isinstance(prediction["alphas"], list) else prediction["alphas"]
                    
                    all_predictions.append(pred_dict)
                
                # Save intermediate results every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    self._save_intermediate_results(all_predictions, batch_idx + 1)
        
        # Compute final metrics
        self.metrics.add_batch(all_predictions)
        final_metrics = self.metrics.compute_metrics()
        
        # Save results
        self._save_results(all_predictions, final_metrics)
        
        # Log to WandB
        if not self.config.get("debug", True):
            wandb.log(final_metrics)
            wandb.finish()
        
        return final_metrics
    
    def _save_intermediate_results(self, predictions: List[Dict], batch_count: int):
        intermediate_file = os.path.join(
            self.output_dir, 
            f"intermediate_results_batch_{batch_count}.json"
        )
        with open(intermediate_file, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    def _save_results(self, predictions: List[Dict], metrics: Dict[str, float]):
        # Save predictions
        if self.config["evaluation"]["save_predictions"]:
            predictions_file = os.path.join(self.output_dir, "predictions.json")
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        detailed_results = self.metrics.get_detailed_results()
        detailed_file = os.path.join(self.output_dir, "detailed_results.json")
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
        print("Final Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Test models on HotpotQA dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/hotpotqa_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Override number of samples to test"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default=None,
        choices=["train", "dev", "test"],
        help="Override dataset split"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply overrides
    if args.output_dir:
        config["evaluation"]["output_dir"] = args.output_dir
    if args.num_samples:
        config["data"]["num_samples"] = args.num_samples
    if args.split:
        config["data"]["split"] = args.split
    
    # Create tester and run
    tester = HotpotQATester(config)
    metrics = tester.test()
    
    return metrics


if __name__ == "__main__":
    main()

