"""
MuSiQue Evaluation Metrics for Multi-hop Question Answering
"""

import re
from typing import List, Dict, Any
import numpy as np


class MuSiQueMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions: List[str] = []
        self.references: List[str] = []
        self.predicted_explanations: List[str] = []
        self.reference_explanations: List[str] = []
    
    def add_batch(self, predictions: List[Dict[str, Any]]):
        for pred in predictions:
            self.predictions.append(pred.get("predicted_answer", ""))
            self.references.append(pred.get("answer", ""))
            self.predicted_explanations.append(pred.get("predicted_explanation", ""))
            self.reference_explanations.append(pred.get("reference_explanation", ""))
    
    def normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        parts = [p.strip() for p in answer.split() if p.strip()]
        return ' '.join(parts)
    
    def exact_match(self, pred: str, ref: str) -> bool:
        return self.normalize_answer(pred) == self.normalize_answer(ref)
    
    def f1_score(self, pred: str, ref: str) -> float:
        pred_norm = self.normalize_answer(pred)
        ref_norm = self.normalize_answer(ref)
        if not pred_norm or not ref_norm:
            return 0.0
        pt = pred_norm.split()
        rt = ref_norm.split()
        pt_set = set(pt)
        rt_set = set(rt)
        if not pt_set or not rt_set:
            return 0.0
        common = pt_set.intersection(rt_set)
        precision = len(common) / len(pt_set) if pt_set else 0.0
        recall = len(common) / len(rt_set) if rt_set else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def extract_answer_from_response(self, response: str) -> str:
        patterns = [
            r'(?:final\s+)?answer\s*:\s*([^\n]+)',
            r'the\s+answer\s+is\s*[:]?\s*([^\n]+)',
            r'(?:final\s+)?answer:\s*(.+?)(?:\n|$)',
            r'answer:\s*(.+?)(?:\n|$)'
        ]
        for p in patterns:
            m = re.search(p, response, re.IGNORECASE | re.DOTALL)
            if m:
                a = m.group(1).strip()
                a = re.sub(r'^\W+|\W+$', '', a)
                if a:
                    return a
        return response.strip()
    
    def compute_metrics(self) -> Dict[str, float]:
        if not self.predictions:
            return {}
        exact = 0
        f1s: List[float] = []
        for pred, ref in zip(self.predictions, self.references):
            if isinstance(pred, str) and len(pred) > 100:
                ans = self.extract_answer_from_response(pred)
            else:
                ans = pred
            if self.exact_match(ans, ref):
                exact += 1
            f1s.append(self.f1_score(ans, ref))
        total = len(self.predictions)
        return {
            "exact_match": exact / total if total else 0.0,
            "f1_score": float(np.mean(f1s)) if f1s else 0.0,
            "total_samples": total,
        }
    
    def get_detailed_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i, (pred, ref) in enumerate(zip(self.predictions, self.references)):
            if isinstance(pred, str) and len(pred) > 100:
                ans = self.extract_answer_from_response(pred)
            else:
                ans = pred
            results.append({
                "sample_id": i,
                "predicted_answer": ans,
                "reference_answer": ref,
                "exact_match": self.exact_match(ans, ref),
                "f1_score": self.f1_score(ans, ref),
            })
        return results


