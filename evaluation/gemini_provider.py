"""
evaluation/gemini_provider.py
------------------------------
Google Gemini Flash API provider for downstream context evaluation and LLM-as-a-Judge assessment.

Supports:
  - Gemini 1.5 Flash / Gemini 2.0 Flash
  - Fallback to mock generation when API key is unconfigured or during offline testing
  - Token accounting (prompt & completion)
  - Latency measurement (seconds)
  - Cost tracking ($0.075 / 1M input tokens, $0.30 / 1M output tokens for Gemini Flash)
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Standard Gemini Flash pricing (per 1M tokens)
GEMINI_FLASH_INPUT_COST_PER_M  = 0.075 / 1_000_000.0
GEMINI_FLASH_OUTPUT_COST_PER_M = 0.300 / 1_000_000.0


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    estimated_cost_usd: float
    model_name: str


class BaseLLMProvider:
    """Abstract base class for downstream LLM providers."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


class GeminiFlashProvider(BaseLLMProvider):
    """
    Official Provider for Google Gemini Flash models.
    Uses `google-generativeai` if installed and GEMINI_API_KEY / GOOGLE_API_KEY is available.
    Falls back gracefully to deterministic response simulation if key is missing or SDK uninstalled.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._sdk_available = False
        self._genai_model = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai_model = genai.GenerativeModel(self.model_name)
                self._sdk_available = True
                logger.info(f"GeminiFlashProvider initialized with model: {self.model_name}")
            except ImportError:
                logger.warning("google-generativeai SDK not installed. Operating in fallback mode.")
            except Exception as e:
                logger.warning(f"Failed to initialize google.generativeai: {e}. Operating in fallback mode.")

    @property
    def name(self) -> str:
        return f"GeminiFlash ({self.model_name})"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.time()

        # Offline / Fallback mode
        if not self._sdk_available or not self._genai_model:
            return self._mock_generate(prompt, t0)

        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            response = self._genai_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            elapsed = time.time() - t0

            response_text = response.text if hasattr(response, "text") else str(response)

            # Extract token counts if usage_metadata available
            prompt_tokens = len(full_prompt.split()) * 4 // 3  # rough estimate default
            completion_tokens = len(response_text.split()) * 4 // 3

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", prompt_tokens)
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", completion_tokens)

            cost = (prompt_tokens * GEMINI_FLASH_INPUT_COST_PER_M) + (completion_tokens * GEMINI_FLASH_OUTPUT_COST_PER_M)

            return LLMResponse(
                text=response_text.strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_sec=round(elapsed, 4),
                estimated_cost_usd=round(cost, 6),
                model_name=self.model_name,
            )

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}. Falling back to deterministic simulation.")
            return self._mock_generate(prompt, t0)

    def _mock_generate(self, prompt: str, start_time: float) -> LLMResponse:
        """Deterministic mock generator for offline benchmarking & unit testing."""
        elapsed = time.time() - start_time
        prompt_words = prompt.split()
        prompt_tokens = int(len(prompt_words) * 1.3) + 5
        
        mock_reply = "Based on the provided contract context, the clause specifies standard terms and conditions."
        completion_tokens = int(len(mock_reply.split()) * 1.3)
        cost = (prompt_tokens * GEMINI_FLASH_INPUT_COST_PER_M) + (completion_tokens * GEMINI_FLASH_OUTPUT_COST_PER_M)

        return LLMResponse(
            text=mock_reply,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_sec=max(0.001, round(elapsed, 4)),
            estimated_cost_usd=round(cost, 6),
            model_name=f"{self.model_name}-mock",
        )
