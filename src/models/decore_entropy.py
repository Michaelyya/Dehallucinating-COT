from typing import List, Optional, Tuple
from contextlib import contextmanager

import copy
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel
from src.utils.modelling_llama import LlamaForCausalLM


class DeCoReEntropy(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)
        
        # Check if model supports native block_list
        self.supports_block_list = getattr(self, 'supports_block_list', True)
        if not self.supports_block_list:
            print(f"[DeCoReEntropy] Model does not support native block_list, will use hooks for head ablation")

        if decoder_configs.configs.amateur_model_name_or_path is not None:
            if "llama" in decoder_configs.configs.amateur_model_name_or_path.lower():
                self.amateur_model = LlamaForCausalLM.from_pretrained(
                    decoder_configs.configs.amateur_model_name_or_path,
                    use_flash_attention_2="flash_attention_2",
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()
                self.amateur_attn_mode = "flash"
            else:
                raise NotImplementedError(
                    "Amateur model other than Llama3-8b-Instruct is not supported yet"
                )

            self.amateur_tokenizer = AutoTokenizer.from_pretrained(
                decoder_configs.configs.amateur_model_name_or_path
            )

            self._load_retrieval_heads(
                decoder_configs.configs.amateur_model_name_or_path
            )
        else:
            self.amateur_model = None
            self._load_retrieval_heads(model_configs.configs.model_name_or_path)

        print("Retrieval heads: ", self.retrieval_heads)

        self.alpha_cap = decoder_configs.configs.get("alpha_cap", None)

        self.scale_alpha = decoder_configs.configs.get("scale_alpha", False)
        
        # For hook-based ablation
        self._ablation_hooks = []

    def _get_attention_layers(self):
        """Get the attention layers from the model for hook registration."""
        layers = []
        # Try different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama/Mistral/Qwen style
            for layer in self.model.model.layers:
                if hasattr(layer, 'self_attn'):
                    layers.append(layer.self_attn)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            for layer in self.model.transformer.h:
                if hasattr(layer, 'attn'):
                    layers.append(layer.attn)
        return layers

    def _create_ablation_hook(self, head_idx, num_heads):
        """Create a hook that zeros out a specific attention head."""
        def hook_fn(module, args, output):
            # Handle different output formats
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None
            
            # Get dimensions - attn_output is typically (batch, seq, hidden)
            if len(attn_output.shape) == 3:
                batch_size, seq_len, hidden_dim = attn_output.shape
                head_dim = hidden_dim // num_heads
                
                # Reshape, zero out head, reshape back
                attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
                attn_output = attn_output.clone()  # Avoid in-place modification
                attn_output[:, :, head_idx, :] = 0
                attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
                
                if rest is not None:
                    return (attn_output,) + rest
                return attn_output
            return output
        return hook_fn

    @contextmanager
    def _ablate_heads_with_hooks(self):
        """Context manager to ablate retrieval heads using forward hooks."""
        hooks = []
        try:
            layers = self._get_attention_layers()
            num_heads = self.model.config.num_attention_heads
            
            for layer_idx, head_idx in self.retrieval_heads:
                if layer_idx < len(layers):
                    hook = layers[layer_idx].register_forward_hook(
                        self._create_ablation_hook(head_idx, num_heads)
                    )
                    hooks.append(hook)
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def _model_forward_with_ablation(self, input_ids, past_key_values, use_cache, attn_mode=None):
        """Forward pass with head ablation - uses hooks if block_list not supported."""
        if self.supports_block_list:
            # Use native block_list support
            return self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                attn_mode=attn_mode,
                block_list=self.retrieval_heads,
            )
        else:
            # Use hooks for ablation
            with self._ablate_heads_with_hooks():
                return self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

    def _load_retrieval_heads(self, model_name_or_path):
        print(f"Loading retrieval heads {model_name_or_path}")
        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads

        model_base_name = model_name_or_path.split("/")[1]
        file_path = os.path.join(
            self.decoder_configs.configs.retrieval_heads_dir,
            f"{model_base_name}.json",
        )
        # Convert to absolute path for clarity
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        print(f"Loading from: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        if os.path.exists(file_path):
            print(f"File size: {os.path.getsize(file_path)} bytes")
            # Read first 200 chars to verify file content
            with open(file_path, 'r') as f:
                first_chars = f.read(200)
                print(f"First 200 chars of file: {first_chars}")

        with open(file_path) as file:
            head_list = json.loads(file.readline())

        print(f"Loaded {len(head_list)} head entries from file")
        print(f"First few entries: {list(head_list.items())[:3]}")
        
        stable_block_list = [(l[0], np.mean(l[1])) for l in head_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        print(f"Top {self.num_retrieval_heads} heads after sorting: {stable_block_list[:self.num_retrieval_heads]}")
        
        self.retrieval_heads = [
            [int(ll) for ll in l[0].split("-")] for l in stable_block_list
        ][: self.num_retrieval_heads]

    def _calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if self.scale_alpha:
            entropy = entropy / np.log(probs.shape[-1])

        return entropy

    def generate_self_contrast(self, inputs, return_attentions: bool = False) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReEntropy"
        self.model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = tokenised_inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            hallucinated_past_kv = copy.deepcopy(input_logits.past_key_values)
            alphas = []
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                # Base model forward (no ablation)
                if self.supports_block_list:
                    base_outputs = self.model(
                        input_ids=last_input_token,
                        past_key_values=base_past_kv,
                        use_cache=True,
                        attn_mode=self.attn_mode,
                    )
                else:
                    base_outputs = self.model(
                        input_ids=last_input_token,
                        past_key_values=base_past_kv,
                        use_cache=True,
                    )
                
                # Hallucinated model forward (with head ablation)
                hallucinated_outputs = self._model_forward_with_ablation(
                    input_ids=last_input_token,
                    past_key_values=hallucinated_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode if self.supports_block_list else None,
                )

                base_past_kv = base_outputs.past_key_values
                hallucinated_past_kv = hallucinated_outputs.past_key_values

                alpha = self._calculate_entropy(base_outputs.logits[0, -1])

                alphas += [alpha.item()]

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha with the entropy cap
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                base_logits = base_outputs.logits[0, -1]
                base_logits = base_logits.log_softmax(dim=-1)
                hallucinated_logits = hallucinated_outputs.logits[0, -1]
                hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)

                next_token_logits = (
                    1 + alpha
                ) * base_logits - alpha * hallucinated_logits

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": alphas}

    def generate_amateur_contrast(
        self, inputs, return_attentions: bool = False
    ) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReEntropy"

        assert self.amateur_model is not None, "Amateur model not loaded"

        self.model.eval()
        self.amateur_model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        expert_tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        amateur_tokenised_inputs = self._verbalise_input(
            prompt,
            use_system_prompt=use_system_prompt,
            tokenizer=self.amateur_tokenizer,
        ).to(self.model.device)

        # Predict
        with torch.inference_mode():
            expert_input_logits = self.model(
                input_ids=expert_tokenised_inputs[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            amateur_input_logits = self.amateur_model(
                input_ids=amateur_tokenised_inputs[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            generated_ids = []
            last_input_token = expert_tokenised_inputs[:, -1]
            expert_past_kv = copy.deepcopy(expert_input_logits.past_key_values)
            amateur_past_kv = copy.deepcopy(amateur_input_logits.past_key_values)
            alphas = []
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                expert_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=expert_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )
                amateur_outputs = self.amateur_model(
                    input_ids=last_input_token,
                    past_key_values=amateur_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                    block_list=self.retrieval_heads,
                )

                expert_past_kv = expert_outputs.past_key_values
                amateur_past_kv = amateur_outputs.past_key_values

                alpha = self._calculate_entropy(expert_outputs.logits[0, -1])

                alphas += [alpha.item()]

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha with the entropy cap
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                expert_logits = expert_outputs.logits[0, -1]
                expert_logits = expert_logits.log_softmax(dim=-1)
                amateur_logits = amateur_outputs.logits[0, -1]
                amateur_logits = amateur_logits.log_softmax(dim=-1)

                next_token_logits = (1 + alpha) * expert_logits - alpha * amateur_logits

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": alphas}

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        if self.amateur_model is not None:
            return self.generate_amateur_contrast(inputs, return_attentions)
        else:
            return self.generate_self_contrast(inputs, return_attentions)

    def lm_score_self_contrast(
        self,
        prompt,
        answer,
    ):
        prompted_question = prompt["prompted_question"][0]

        # Only relevant for instruct model
        if len(prompt["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        with torch.no_grad():
            if type(prompted_question) == list:
                input_text = prompted_question + [answer]
            else:
                input_text = prompted_question + answer

            input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            # Base model forward
            if self.supports_block_list:
                base_outputs = self.model(input_ids, attn_mode=self.attn_mode)[0]
                hallucinated_outputs = self.model(
                    input_ids, block_list=self.retrieval_heads, attn_mode=self.attn_mode
                )[0]
            else:
                base_outputs = self.model(input_ids)[0]
                with self._ablate_heads_with_hooks():
                    hallucinated_outputs = self.model(input_ids)[0]

            base_logits = base_outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            hallucinated_logits = hallucinated_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]

            entropies = []
            for i in range(base_logits.shape[0]):
                entropies += [self._calculate_entropy(base_logits[i, :])]

            alpha = torch.stack(entropies).unsqueeze(1)

            if self.alpha_cap:
                # If the entropy is too high, cap the alpha with the entropy cap
                alpha = torch.min(alpha, torch.tensor(self.alpha_cap).to(alpha.device))

            base_logits = base_logits.log_softmax(dim=-1)
            hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)

            diff_logits = (1 + alpha) * base_logits - alpha * hallucinated_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs

    def lm_score_amateur_contrast(
        self,
        prompt,
        answer,
    ):
        prompted_question = prompt["prompted_question"][0]

        # Only relevant for instruct model
        if len(prompt["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        with torch.no_grad():
            if type(prompted_question) == list:
                input_text = prompted_question + [answer]
            else:
                input_text = prompted_question + answer

            expert_input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            expert_prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = expert_input_ids[0, expert_prefix_ids.shape[-1] :]

            amateur_input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
                tokenizer=self.amateur_tokenizer,
            ).to(self.amateur_model.device)
            amateur_prefix_ids = self._verbalise_input(
                prompted_question,
                use_system_prompt=use_system_prompt,
                tokenizer=self.amateur_tokenizer,
            ).to(self.amateur_model.device)

            expert_outputs = self.model(expert_input_ids, attn_mode=self.attn_mode)[0]
            amateur_outputs = self.amateur_model(
                amateur_input_ids,
                block_list=self.retrieval_heads,
                attn_mode=self.amateur_attn_mode,
            )[0]

            expert_logits = expert_outputs[0, expert_prefix_ids.shape[-1] - 1 : -1, :]
            amateur_logits = amateur_outputs[
                0, amateur_prefix_ids.shape[-1] - 1 : -1, :
            ]

            entropies = []
            for i in range(expert_logits.shape[0]):
                entropies += [self._calculate_entropy(expert_logits[i, :])]

            alpha = torch.stack(entropies).unsqueeze(1)

            if self.alpha_cap:
                # If the entropy is too high, cap the alpha with the entropy cap
                alpha = torch.min(alpha, torch.tensor(self.alpha_cap).to(alpha.device))

            expert_logits = expert_logits.log_softmax(dim=-1)
            amateur_logits = amateur_logits.log_softmax(dim=-1)

            diff_logits = (1 + alpha) * expert_logits - alpha * amateur_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs

    def lm_score(
        self,
        prompt,
        answer,
    ):
        if self.amateur_model is not None:
            return self.lm_score_amateur_contrast(prompt, answer)
        else:
            return self.lm_score_self_contrast(prompt, answer)
