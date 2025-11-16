"""
Integration helpers for running benchmarks with reasoning head masking.
"""

import os
import yaml
import json
from typing import List, Tuple, Dict, Any, Optional
import copy


def create_masked_config(
    base_config_path: str,
    masked_heads: List[Tuple[int, int]],
    output_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a config file with reasoning heads masked.
    
    This modifies the decoder config to use a baseline method that masks
    the specified heads during generation.
    
    Args:
        base_config_path: Path to base config file
        masked_heads: List of (layer, head) tuples to mask
        output_config_path: Optional path to save modified config
    
    Returns:
        Modified config dictionary
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a copy
    config = copy.deepcopy(config)
    
    # Store masked heads in config
    config["masked_heads"] = masked_heads
    
    # Modify decoder to use baseline with masking
    # We'll use BaselineMaskedRetrievalHead or create a custom method
    if "decoder" in config:
        # Store original decoder config
        config["original_decoder"] = copy.deepcopy(config["decoder"])
        
        # Modify to use baseline with masked heads
        config["decoder"]["name"] = "BaselineMaskedReasoningHeads"
        config["decoder"]["method"] = "BaselineMaskedReasoningHeads"
        config["decoder"]["configs"] = {
            "masked_heads": masked_heads,
            "num_retrieval_heads": len(masked_heads)
        }
    
    # Save if output path provided
    if output_config_path:
        with open(output_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved masked config to {output_config_path}")
    
    return config


def create_reasoning_head_decoder_class():
    """
    Create a custom decoder class that masks reasoning heads.
    
    This can be registered with the model factory to use discovered reasoning heads.
    """
    # This would be a new model class similar to BaselineMaskedRetrievalHead
    # For now, we'll provide instructions in the README
    pass


def prepare_benchmark_configs(
    discovered_heads_file: str,
    base_configs: Dict[str, str],
    output_dir: str = "./masked_configs"
) -> Dict[str, Dict[str, str]]:
    """
    Prepare benchmark configs with reasoning head masking.
    
    Args:
        discovered_heads_file: Path to discovered heads JSON file
        base_configs: Dict mapping benchmark names to config paths
        output_dir: Directory to save modified configs
    
    Returns:
        Dict with "main" and "baseline" config paths for each benchmark
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load discovered heads
    with open(discovered_heads_file, 'r') as f:
        heads_data = json.load(f)
    
    heads = heads_data.get("heads", [])
    
    # Convert to (layer, head) format
    masked_heads = [(h["layer"], h["head"]) for h in heads]
    
    prepared_configs = {}
    
    for benchmark_name, base_config_path in base_configs.items():
        # Create masked config
        masked_config_path = os.path.join(
            output_dir,
            f"{benchmark_name}_masked.yaml"
        )
        
        create_masked_config(
            base_config_path,
            masked_heads,
            masked_config_path
        )
        
        # Find baseline config (assume naming convention)
        baseline_config_path = base_config_path.replace("_model_", "_baseline_")
        if not os.path.exists(baseline_config_path):
            # Try alternative naming
            baseline_config_path = base_config_path.replace("model", "baseline")
        
        prepared_configs[benchmark_name] = {
            "main": masked_config_path,
            "baseline": baseline_config_path if os.path.exists(baseline_config_path) else base_config_path
        }
    
    return prepared_configs


def get_masked_heads_from_config(config: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
    """Extract masked heads from config."""
    if "masked_heads" in config:
        return config["masked_heads"]
    if "decoder" in config and "configs" in config["decoder"]:
        decoder_configs = config["decoder"]["configs"]
        if "masked_heads" in decoder_configs:
            return decoder_configs["masked_heads"]
    return None


def create_custom_decoder_model():
    """
    Instructions for creating a custom decoder model class.
    
    To properly integrate reasoning head masking, you would need to:
    
    1. Create a new model class (e.g., ReasoningHeadMaskedModel) that extends BaseModel
    2. In the generate method, pass block_list=masked_heads to model calls
    3. Register it in the factory
    
    Example:
    
    class ReasoningHeadMaskedModel(BaseModel):
        def __init__(self, model_configs, decoder_configs):
            super().__init__(model_configs, decoder_configs)
            self.masked_heads = decoder_configs.configs.get("masked_heads", [])
        
        def generate(self, inputs, return_attentions=False):
            # Use block_list parameter to mask heads
            outputs = self.model(
                input_ids=inputs,
                block_list=self.masked_heads,
                ...
            )
            ...
    """
    pass

