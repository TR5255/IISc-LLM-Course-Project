from typing import Dict, Any

class DownstreamLLMEvaluator:
    """Evaluates downstream LLM correctness on compressed context versus original context.

    Initially mocks answering to allow local evaluations of routers and policies without api keys.
    """
    def __init__(self, downstream_model_name: str = "mock-gpt-4o"):
        self.downstream_model_name = downstream_model_name

    def evaluate_answer_possibility(
        self,
        question: str,
        compressed_text: str,
        reference_answer: str
    ) -> Dict[str, Any]:
        """Simulates downstream LLM response generation based on the compressed context.

        Uses simple keyword overlap heuristics between compressed text and reference answer.
        """
        ref_tokens = set(reference_answer.lower().replace(",", "").replace(".", "").split())
        
        if not ref_tokens:
            return {
                "downstream_model": self.downstream_model_name,
                "accuracy_score": 1.0,
                "explanation": "No reference answer tokens to match; defaults to correct."
            }
            
        text_lower = compressed_text.lower()
        matched = sum(1 for token in ref_tokens if token in text_lower)
        recall = matched / len(ref_tokens)
        
        # Simple threshold: if 50% of reference answer keywords exist in compressed text,
        # we consider the downstream LLM capable of answering it correctly.
        accuracy = 1.0 if recall >= 0.5 else 0.0
        
        return {
            "downstream_model": self.downstream_model_name,
            "accuracy_score": accuracy,
            "keyword_recall": recall,
            "explanation": f"Matched {matched}/{len(ref_tokens)} reference keywords."
        }
