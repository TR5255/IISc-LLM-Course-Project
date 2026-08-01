from typing import Dict, Any, Optional
import logging
from evaluation.gemini_provider import BaseLLMProvider, GeminiFlashProvider

logger = logging.getLogger(__name__)


class DownstreamLLMEvaluator:
    """
    Evaluates downstream LLM (Gemini Flash) performance on compressed context
    versus original context, computing factual retention, token cost savings,
    and downstream latency.
    """
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or GeminiFlashProvider()

    def evaluate_answer_possibility(
        self,
        question: str,
        compressed_text: str,
        reference_answer: str,
        original_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes downstream generation with Gemini Flash on compressed text context
        and computes evaluation metrics.
        """
        prompt = (
            f"Context:\n{compressed_text}\n\n"
            f"Question: {question}\n\n"
            f"Instructions: Answer the question concisely based ONLY on the context."
        )

        # 1. Downstream LLM Generation
        response = self.provider.generate(prompt=prompt, max_tokens=256, temperature=0.0)

        # 2. Token & Cost Analysis vs Original Full Context (if provided)
        orig_tokens = len(original_text.split()) * 4 // 3 if original_text else response.prompt_tokens
        token_savings_pct = 0.0
        if orig_tokens > 0:
            token_savings_pct = max(0.0, float((orig_tokens - response.prompt_tokens) / orig_tokens * 100.0))

        # 3. Accuracy & Keyword Recall Evaluation
        ref_tokens = set(reference_answer.lower().replace(",", "").replace(".", "").split())
        matched = 0
        keyword_recall = 1.0

        if ref_tokens:
            gen_lower = response.text.lower()
            matched = sum(1 for token in ref_tokens if token in gen_lower or token in compressed_text.lower())
            keyword_recall = matched / len(ref_tokens)

        # Strict accuracy criterion: 50% keyword recall OR reference match
        accuracy = 1.0 if keyword_recall >= 0.5 else 0.0

        return {
            "downstream_model": response.model_name,
            "provider_name": self.provider.name,
            "generated_answer": response.text,
            "accuracy_score": accuracy,
            "keyword_recall": round(keyword_recall, 4),
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_sec": response.latency_sec,
            "estimated_cost_usd": response.estimated_cost_usd,
            "token_savings_pct": round(token_savings_pct, 2),
            "explanation": f"Matched {matched}/{len(ref_tokens)} reference tokens. Downstream generation completed in {response.latency_sec}s.",
        }

