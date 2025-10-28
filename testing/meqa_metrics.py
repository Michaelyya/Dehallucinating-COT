"""
MEQA Evaluation Metrics for Multi-hop Event-centric Question Answering
"""

import re
import json
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict


class MEQAMetrics:
    """
    Evaluation metrics for MEQA dataset focusing on:
    1. Answer accuracy (exact match and F1)
    2. Reasoning quality (explanation faithfulness)
    3. Multi-hop reasoning capability
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.references = []
        self.explanations = []
        self.reference_explanations = []
    
    def add_batch(self, predictions: List[Dict[str, Any]]):
        """Add a batch of predictions for evaluation"""
        for pred in predictions:
            self.predictions.append(pred.get("predicted_answer", ""))
            self.references.append(pred.get("answer", ""))
            self.explanations.append(pred.get("predicted_explanation", ""))
            self.reference_explanations.append(pred.get("reference_explanation", ""))
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        
        # Convert to lowercase and strip whitespace
        answer = answer.lower().strip()
        
        # Remove extra punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Split by common separators and sort for consistent comparison
        parts = re.split(r'[,;]', answer)
        parts = [part.strip() for part in parts if part.strip()]
        parts = sorted(list(set(parts)))  # Remove duplicates and sort
        
        return ', '.join(parts)
    
    def exact_match(self, pred: str, ref: str) -> bool:
        """Check if prediction exactly matches reference"""
        pred_norm = self.normalize_answer(pred)
        ref_norm = self.normalize_answer(ref)
        return pred_norm == ref_norm
    
    def f1_score(self, pred: str, ref: str) -> float:
        """Calculate F1 score between prediction and reference"""
        pred_norm = self.normalize_answer(pred)
        ref_norm = self.normalize_answer(ref)
        
        if not pred_norm or not ref_norm:
            return 0.0
        
        # Split into words/tokens
        pred_tokens = set(pred_norm.split())
        ref_tokens = set(ref_norm.split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Calculate precision and recall
        common_tokens = pred_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the final answer from a CoT response"""
        # Look for patterns like "Answer:", "Final answer:", etc.
        answer_patterns = [
            r'(?:final\s+)?answer:\s*(.+?)(?:\n|$)',
            r'answer:\s*(.+?)(?:\n|$)',
            r'the\s+answer\s+is:\s*(.+?)(?:\n|$)',
            r'answer\s+is:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, try to extract the last sentence or phrase
        sentences = response.split('.')
        if sentences:
            last_sentence = sentences[-1].strip()
            if last_sentence and len(last_sentence) > 3:
                return last_sentence
        
        return response.strip()
    
    def extract_explanation_from_response(self, response: str) -> str:
        """Extract the reasoning explanation from a CoT response"""
        # Remove the final answer part to get the explanation
        answer_patterns = [
            r'(?:final\s+)?answer:\s*.+',
            r'answer:\s*.+',
            r'the\s+answer\s+is:\s*.+',
            r'answer\s+is:\s*.+',
        ]
        
        explanation = response
        for pattern in answer_patterns:
            explanation = re.sub(pattern, '', explanation, flags=re.IGNORECASE | re.DOTALL)
        
        return explanation.strip()
    
    def evaluate_reasoning_quality(self, pred_explanation: str, ref_explanation: str) -> Dict[str, float]:
        """Evaluate the quality of reasoning explanations"""
        if not pred_explanation or not ref_explanation:
            return {"explanation_f1": 0.0, "explanation_bleu": 0.0}
        
        # Simple F1 score on explanation tokens
        pred_tokens = set(pred_explanation.lower().split())
        ref_tokens = set(ref_explanation.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return {"explanation_f1": 0.0, "explanation_bleu": 0.0}
        
        common_tokens = pred_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
        
        explanation_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simple BLEU-like score (n-gram overlap)
        explanation_bleu = self._calculate_bleu_score(pred_explanation, ref_explanation)
        
        return {
            "explanation_f1": explanation_f1,
            "explanation_bleu": explanation_bleu
        }
    
    def _calculate_bleu_score(self, pred: str, ref: str, n: int = 2) -> float:
        """Calculate a simple BLEU-like score"""
        def get_ngrams(text: str, n: int) -> List[str]:
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        pred_ngrams = get_ngrams(pred, n)
        ref_ngrams = get_ngrams(ref, n)
        
        if not pred_ngrams or not ref_ngrams:
            return 0.0
        
        # Calculate precision for each n-gram
        precision_scores = []
        for i in range(1, n+1):
            pred_i_grams = get_ngrams(pred, i)
            ref_i_grams = get_ngrams(ref, i)
            
            if not pred_i_grams:
                precision_scores.append(0.0)
                continue
            
            matches = sum(1 for gram in pred_i_grams if gram in ref_i_grams)
            precision_scores.append(matches / len(pred_i_grams))
        
        # Calculate geometric mean
        if all(score > 0 for score in precision_scores):
            return np.exp(np.mean([np.log(score) for score in precision_scores]))
        else:
            return 0.0
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        if not self.predictions:
            return {}
        
        # Initialize metrics
        exact_matches = 0
        f1_scores = []
        explanation_f1_scores = []
        explanation_bleu_scores = []
        
        # Process each prediction
        for i, (pred, ref, pred_exp, ref_exp) in enumerate(
            zip(self.predictions, self.references, self.explanations, self.reference_explanations)
        ):
            # Extract answer from response if needed
            if isinstance(pred, str) and len(pred) > 100:  # Likely a full response
                extracted_answer = self.extract_answer_from_response(pred)
                extracted_explanation = self.extract_explanation_from_response(pred)
            else:
                extracted_answer = pred
                extracted_explanation = pred_exp
            
            # Calculate answer metrics
            if self.exact_match(extracted_answer, ref):
                exact_matches += 1
            
            f1 = self.f1_score(extracted_answer, ref)
            f1_scores.append(f1)
            
            # Calculate explanation metrics
            exp_metrics = self.evaluate_reasoning_quality(extracted_explanation, ref_exp)
            explanation_f1_scores.append(exp_metrics["explanation_f1"])
            explanation_bleu_scores.append(exp_metrics["explanation_bleu"])
        
        # Calculate final metrics
        total = len(self.predictions)
        metrics = {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "f1_score": np.mean(f1_scores) if f1_scores else 0.0,
            "explanation_f1": np.mean(explanation_f1_scores) if explanation_f1_scores else 0.0,
            "explanation_bleu": np.mean(explanation_bleu_scores) if explanation_bleu_scores else 0.0,
            "total_samples": total,
        }
        
        return metrics
    
    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """Get detailed results for each sample"""
        results = []
        
        for i, (pred, ref, pred_exp, ref_exp) in enumerate(
            zip(self.predictions, self.references, self.explanations, self.reference_explanations)
        ):
            # Extract answer from response if needed
            if isinstance(pred, str) and len(pred) > 100:  # Likely a full response
                extracted_answer = self.extract_answer_from_response(pred)
                extracted_explanation = self.extract_explanation_from_response(pred)
            else:
                extracted_answer = pred
                extracted_explanation = pred_exp
            
            # Calculate metrics for this sample
            exact_match = self.exact_match(extracted_answer, ref)
            f1 = self.f1_score(extracted_answer, ref)
            exp_metrics = self.evaluate_reasoning_quality(extracted_explanation, ref_exp)
            
            results.append({
                "sample_id": i,
                "predicted_answer": extracted_answer,
                "reference_answer": ref,
                "predicted_explanation": extracted_explanation,
                "reference_explanation": ref_exp,
                "exact_match": exact_match,
                "f1_score": f1,
                "explanation_f1": exp_metrics["explanation_f1"],
                "explanation_bleu": exp_metrics["explanation_bleu"],
            })
        
        return results


def evaluate_meqa_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate MEQA predictions and return metrics
    
    Args:
        predictions: List of prediction dictionaries with keys:
            - predicted_answer: Model's predicted answer
            - answer: Ground truth answer
            - predicted_explanation: Model's reasoning explanation
            - reference_explanation: Ground truth explanation
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = MEQAMetrics()
    metrics.add_batch(predictions)
    return metrics.compute_metrics()


if __name__ == "__main__":
    # Test the metrics
    test_predictions = [
        {
            "predicted_answer": "drone",
            "answer": "drone",
            "predicted_explanation": "The question asks what was destroyed when a drone was taken down. Looking at the context, it mentions that drones were shot down.",
            "reference_explanation": "What event contains drone is the Artifact? shot down@473 What event when #1 has a Artifact? shot down@473 what was destroyed in the #2? drone"
        },
        {
            "predicted_answer": "vehicle, car",
            "answer": "vehicle",
            "predicted_explanation": "The context mentions that a vehicle was destroyed in an airstrike.",
            "reference_explanation": "What event contains the car is the Target? struck@315 What event is after #1 has a Artifact? destroyed@274 What was destroyed in the #2? vehicle"
        }
    ]
    
    metrics = evaluate_meqa_predictions(test_predictions)
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
