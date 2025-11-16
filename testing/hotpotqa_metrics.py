"""
HotpotQA Evaluation Metrics for Multi-hop Question Answering
"""

import re
import json
from typing import List, Dict, Any
import numpy as np


class HotpotQAMetrics:
    """
    Evaluation metrics for HotpotQA dataset focusing on:
    1. Answer accuracy (exact match and F1)
    2. Supporting facts accuracy (F1)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.references = []
        self.predicted_supporting_facts = []
        self.reference_supporting_facts = []
    
    def add_batch(self, predictions: List[Dict[str, Any]]):
        for pred in predictions:
            self.predictions.append(pred.get("predicted_answer", ""))
            self.references.append(pred.get("answer", ""))
            self.predicted_supporting_facts.append(pred.get("predicted_supporting_facts", []))
            self.reference_supporting_facts.append(pred.get("supporting_facts", []))
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        
        parts = re.split(r'[,;]', answer)
        parts = [part.strip() for part in parts if part.strip()]
        parts = sorted(list(set(parts)))
        
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
        
        pred_tokens = set(pred_norm.split())
        ref_tokens = set(ref_norm.split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common_tokens = pred_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def supporting_facts_f1(self, pred_facts: List[List], ref_facts: List[List]) -> float:
        """Calculate F1 score for supporting facts"""
        if not ref_facts:
            return 1.0 if not pred_facts else 0.0
        
        if not pred_facts:
            return 0.0
        
        # Convert to sets of tuples for comparison
        pred_set = set((fact[0], fact[1]) for fact in pred_facts)
        ref_set = set((fact[0], fact[1]) for fact in ref_facts)
        
        if not pred_set or not ref_set:
            return 0.0
        
        common = pred_set.intersection(ref_set)
        precision = len(common) / len(pred_set) if pred_set else 0.0
        recall = len(common) / len(ref_set) if ref_set else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the final answer from a CoT response"""
        patterns = [
            r'(?:final\s+)?answer\s*:\s*([^\n]+)',
            r'the\s+answer\s+is\s*[:]?\s*([^\n]+)',
            r'(?:final\s+)?answer:\s*(.+?)(?:\n|$)',
            r'answer:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'^\W+|\W+$', '', answer)
                if answer:
                    return answer
        
        return response.strip()
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        if not self.predictions:
            return {}
        
        exact_matches = 0
        f1_scores = []
        sp_f1_scores = []
        
        for i, (pred, ref, pred_sp, ref_sp) in enumerate(
            zip(self.predictions, self.references, 
                self.predicted_supporting_facts, self.reference_supporting_facts)
        ):
            # Extract answer if needed
            if isinstance(pred, str) and len(pred) > 100:
                extracted_answer = self.extract_answer_from_response(pred)
            else:
                extracted_answer = pred
            
            # Calculate answer metrics
            if self.exact_match(extracted_answer, ref):
                exact_matches += 1
            
            f1 = self.f1_score(extracted_answer, ref)
            f1_scores.append(f1)
            
            # Calculate supporting facts metrics
            sp_f1 = self.supporting_facts_f1(pred_sp, ref_sp)
            sp_f1_scores.append(sp_f1)
        
        total = len(self.predictions)
        metrics = {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "f1_score": np.mean(f1_scores) if f1_scores else 0.0,
            "supporting_facts_f1": np.mean(sp_f1_scores) if sp_f1_scores else 0.0,
            "total_samples": total,
        }
        
        return metrics
    
    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """Get detailed results for each sample"""
        results = []
        
        for i, (pred, ref, pred_sp, ref_sp) in enumerate(
            zip(self.predictions, self.references,
                self.predicted_supporting_facts, self.reference_supporting_facts)
        ):
            # Extract answer if needed
            if isinstance(pred, str) and len(pred) > 100:
                extracted_answer = self.extract_answer_from_response(pred)
            else:
                extracted_answer = pred
            
            exact_match = self.exact_match(extracted_answer, ref)
            f1 = self.f1_score(extracted_answer, ref)
            sp_f1 = self.supporting_facts_f1(pred_sp, ref_sp)
            
            results.append({
                "sample_id": i,
                "predicted_answer": extracted_answer,
                "reference_answer": ref,
                "predicted_supporting_facts": pred_sp,
                "reference_supporting_facts": ref_sp,
                "exact_match": exact_match,
                "f1_score": f1,
                "supporting_facts_f1": sp_f1,
            })
        
        return results


def evaluate_hotpotqa_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate HotpotQA predictions and return metrics
    
    Args:
        predictions: List of prediction dictionaries with keys:
            - predicted_answer: Model's predicted answer
            - answer: Ground truth answer
            - predicted_supporting_facts: Model's predicted supporting facts
            - supporting_facts: Ground truth supporting facts
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = HotpotQAMetrics()
    metrics.add_batch(predictions)
    return metrics.compute_metrics()

