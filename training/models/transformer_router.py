"""
training/models/transformer_router.py
---------------------------------------
Wrapper around HuggingFace AutoModelForSequenceClassification and AutoTokenizer for
small neural language model router architectures (Qwen2.5-0.5B, SmolLM, etc.).

Supports forward pass, batch inference predictions, sequence classification head adapter,
and optional LoRA wrapper initialization via PEFT.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TransformerRouterModel:
    """
    HuggingFace Transformer wrapper for router importance classification.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace hub path or local checkpoint path (e.g. 'Qwen/Qwen2.5-0.5B').
    max_length : int
        Maximum token sequence length.
    num_labels : int
        Number of output classification labels (default: 2 for binary relevance).
    device : str
        Device placement ('cpu', 'cuda', 'mps', or 'auto').
    lora_config : dict, optional
        Optional PEFT LoRA configuration dictionary.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        max_length: int = 512,
        num_labels: int = 2,
        device: str = "cpu",
        lora_config: Optional[Dict[str, Any]] = None,
        pretrained_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.num_labels = num_labels
        self.device = device
        self.lora_config = lora_config

        self._tokenizer = tokenizer
        self._model = pretrained_model
        self._is_initialized = (pretrained_model is not None and tokenizer is not None)

    def initialize(self) -> None:
        """Lazy initialization of tokenizer and HuggingFace sequence classification model."""
        if self._is_initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        except ImportError:
            logger.warning("transformers/torch package not installed. Operating in mock mode.")
            self._is_initialized = True
            return

        logger.info("Initializing TransformerRouterModel from: %s", self.model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            trust_remote_code=True,
        )

        # Optional LoRA initialization
        if self.lora_config and self.lora_config.get("enabled", False):
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore
                peft_config = LoraConfig(
                    r=self.lora_config.get("r", 8),
                    lora_alpha=self.lora_config.get("lora_alpha", 16),
                    target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj"]),
                    lora_dropout=self.lora_config.get("lora_dropout", 0.05),
                    bias="none",
                    task_type="SEQ_CLS",
                )
                self._model = get_peft_model(self._model, peft_config)
                logger.info("Applied PEFT LoRA adapter to model.")
            except Exception as err:
                logger.warning("Could not initialize PEFT LoRA adapter: %s", err)

        if self.device != "auto":
            self._model.to(self.device)

        self._is_initialized = True

    def format_input(self, query: str, chunk_text: str) -> str:
        """Format query and chunk into sequence prompt input."""
        return f"[QUERY] {query}\n[CHUNK] {chunk_text}"

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[List[float]]:
        """
        Runs batch prediction on a list of (query, chunk_text) tuples.

        Returns:
            List of raw logit lists (e.g. [[neg_logit, pos_logit], ...]).
        """
        if not pairs:
            return []

        if not self._is_initialized:
            self.initialize()

        texts = [self.format_input(q, c) for q, c in pairs]

        try:
            import torch  # type: ignore
            if hasattr(self._tokenizer, "__call__") and hasattr(self._model, "__call__") and not hasattr(self._tokenizer, "_is_mock"):
                inputs = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                if self.device != "auto" and hasattr(inputs, "to"):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                self._model.eval()
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits.detach().cpu().tolist()
                    return logits
        except ImportError:
            pass

        # Fallback for mock objects in testing or when torch is not installed
        return [[0.0, 1.0] for _ in pairs]
