"""
tests/test_neural_router.py
----------------------------
Unit tests for the neural router model integration framework (Milestone 8).
All tests use mock models or synthetic objects to ensure fast execution without downloading
remote model checkpoints.
"""
from unittest.mock import MagicMock
import pytest
import os

from models.scorer import BaseScorer
from models.model_adapter import ModelAdapter
from models.neural_router import NeuralRouter
from training.models.transformer_router import TransformerRouterModel
from training.transformer_trainer import TransformerRouterTrainer
from training.base_trainer import BaseTrainer


def test_model_adapter_binary_logit():
    """Verify ModelAdapter maps binary logits [neg, pos] to [0.0, 1.0] sigmoid probabilities."""
    adapter = ModelAdapter(output_type="binary_logit")
    
    # Positive logit > Negative logit -> score > 0.5
    score_pos = adapter.adapt_single([0.0, 2.0])
    assert 0.5 < score_pos <= 1.0

    # Negative logit > Positive logit -> score < 0.5
    score_neg = adapter.adapt_single([2.0, 0.0])
    assert 0.0 <= score_neg < 0.5

    # Equal logits -> score == 0.5
    score_eq = adapter.adapt_single([1.0, 1.0])
    assert abs(score_eq - 0.5) < 1e-4


def test_model_adapter_single_logit_and_probability():
    """Verify ModelAdapter handles single_logit and probability output types."""
    adapter_single = ModelAdapter(output_type="single_logit")
    assert 0.0 <= adapter_single.adapt_single(0.0) == 0.5
    assert adapter_single.adapt_single(5.0) > 0.9

    adapter_prob = ModelAdapter(output_type="probability")
    assert adapter_prob.adapt_single(0.85) == 0.85


def test_model_adapter_graded_relevance():
    """Verify ModelAdapter maps float scores to 0-3 graded relevance labels."""
    adapter = ModelAdapter()
    assert adapter.to_graded_relevance(0.1) == 0
    assert adapter.to_graded_relevance(0.35) == 1
    assert adapter.to_graded_relevance(0.60) == 2
    assert adapter.to_graded_relevance(0.90) == 3


def test_neural_router_inherits_basescorer():
    """Verify NeuralRouter is a valid subclass of BaseScorer."""
    router = NeuralRouter()
    assert isinstance(router, BaseScorer)


def test_neural_router_score_with_mock_wrapper():
    """Verify NeuralRouter correctly passes queries to mock wrapper and returns bounded scores."""
    mock_wrapper = MagicMock()
    mock_wrapper.predict_batch.return_value = [[-1.0, 1.0], [2.0, -2.0], [0.0, 0.0]]

    router = NeuralRouter(model_wrapper=mock_wrapper, batch_size=2)
    scores = router.score(
        question="What is the governing law?",
        chunks=["Chunk 1", "Chunk 2", "Chunk 3"]
    )

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert mock_wrapper.predict_batch.call_count == 2  # batch_size=2 -> 2 calls


def test_neural_router_fallback_when_no_wrapper():
    """Verify NeuralRouter fallback behavior when no wrapper is provided."""
    router = NeuralRouter(model_wrapper=None)
    scores = router.score("Query", ["Chunk A", "Chunk B"])
    assert scores == [0.5, 0.5]


def test_transformer_router_model_mock():
    """Verify TransformerRouterModel input formatting and mock prediction."""
    mock_tokenizer = MagicMock()
    mock_tokenizer._is_mock = True
    mock_model = MagicMock()
    
    wrapper = TransformerRouterModel(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        pretrained_model=mock_model,
        tokenizer=mock_tokenizer,
    )
    
    formatted = wrapper.format_input("What is X?", "Y is a concept...")
    assert "[QUERY] What is X?" in formatted
    assert "[CHUNK] Y is a concept..." in formatted

    # Test predict_batch on pre-initialized mock wrapper
    predictions = wrapper.predict_batch([("Query", "Chunk")])
    assert len(predictions) == 1
    assert len(predictions[0]) == 2


def test_transformer_router_trainer_base_trainer_interface():
    """Verify TransformerRouterTrainer inherits BaseTrainer and complies with protocol."""
    trainer = TransformerRouterTrainer()
    assert isinstance(trainer, BaseTrainer)

    # Test fit returns dictionary metrics
    metrics = trainer.fit(train_dataset=None, val_dataset=None)
    required_keys = {"precision", "recall", "f1", "auc", "ndcg@3", "ndcg@5", "recall@1", "recall@3", "recall@5"}
    assert required_keys.issubset(metrics.keys())

    # Test predict_proba
    probs = trainer.predict_proba("Test query", ["Chunk 1", "Chunk 2"])
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_transformer_router_trainer_save_load(tmp_path):
    """Verify TransformerRouterTrainer save and load roundtrip."""
    save_dir = str(tmp_path / "neural_checkpoint")
    trainer = TransformerRouterTrainer()
    trainer.save(save_dir)

    assert os.path.exists(os.path.join(save_dir, "trainer_meta.json"))

    loaded_trainer = TransformerRouterTrainer.load(save_dir)
    assert isinstance(loaded_trainer, TransformerRouterTrainer)
    assert loaded_trainer.model.model_name_or_path == trainer.model.model_name_or_path
