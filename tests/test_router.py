import os
import tempfile
import yaml
from utils.config import load_config
from data.preprocess.splitter import DocumentSplitter
from models.scorer import RandomScorer
from baselines.bm25 import BM25Scorer
from baselines.tfidf import TFIDFScorer
from baselines.embedding import EmbeddingScorer, CharFrequencyScorer
from models.policies import ThresholdPolicy, TopKPolicy
from models.router import SmartAIRouter
from evaluation.retention import calculate_precision_recall, calculate_f1, calculate_ndcg

def test_config_namespace():
    config_data = {
        "model": {"name": "test_model", "settings": {"seed": 10}},
        "policy": {"type": "threshold", "threshold": 0.5}
    }
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
        
    try:
        config = load_config(temp_path)
        assert config.model.name == "test_model"
        assert config.model.settings.seed == 10
        assert config.policy.threshold == 0.5
    finally:
        os.remove(temp_path)

def test_scorers_output_range():
    question = "Who is default?"
    chunks = [
        "First sentence in nda is default.",
        "Second sentence is irrelevant.",
        "Third sentence has default too."
    ]
    
    scorers = [RandomScorer(), BM25Scorer(), TFIDFScorer(), CharFrequencyScorer(), EmbeddingScorer()]
    for scorer in scorers:
        scores = scorer.score(question, chunks)
        assert len(scores) == len(chunks)
        for s in scores:
            assert 0.0 <= s <= 1.0

def test_routing_policies():
    chunks = ["A", "B", "C", "D"]
    scores = [0.1, 0.9, 0.4, 0.7]
    
    thresh_policy = ThresholdPolicy(threshold=0.5)
    selected = thresh_policy.select(chunks, scores)
    # B (0.9) and D (0.7) are >= 0.5
    assert selected == ["B", "D"]

    topk_policy = TopKPolicy(k=2)
    selected = topk_policy.select(chunks, scores)
    # B (0.9) and D (0.7) are the top 2
    assert selected == ["B", "D"]

def test_router_flow():
    scorer = BM25Scorer()
    policy = ThresholdPolicy(threshold=0.1)
    router = SmartAIRouter(scorer=scorer, policy=policy)
    
    question = "NDA term"
    chunks = [
        "NDA agreement governs rules.",
        "Today is a sunny day.",
        "Term is five years."
    ]
    
    selected = router.route(question, chunks)
    assert "NDA agreement governs rules." in selected
    assert "Term is five years." in selected
    assert "Today is a sunny day." not in selected

def test_f1_calculation():
    # Both zero → 0.0
    assert calculate_f1(0.0, 0.0) == 0.0
    # Perfect → 1.0
    assert calculate_f1(1.0, 1.0) == 1.0
    # Partial: P=0.5, R=1.0 → F1=2*(0.5*1.0)/(0.5+1.0) ≈ 0.6667
    f1 = calculate_f1(0.5, 1.0)
    assert abs(f1 - 2/3) < 1e-6
    # One zero → 0.0
    assert calculate_f1(0.0, 1.0) == 0.0
    assert calculate_f1(1.0, 0.0) == 0.0

def test_ndcg_calculation():
    # Chunk relevance scores: {0: 0, 1: 3, 2: 1}
    chunk_scores = {0: 0, 1: 3, 2: 1}

    # Perfect ranking: select chunk 1 (score=3) first, then chunk 2 (score=1)
    ndcg_perfect = calculate_ndcg([1, 2], chunk_scores)
    assert ndcg_perfect > 0.9, f"Perfect NDCG should be close to 1.0, got {ndcg_perfect}"

    # Worst ranking: select chunk 0 (score=0) only
    ndcg_worst = calculate_ndcg([0], chunk_scores)
    assert ndcg_worst == 0.0, f"Selecting only irrelevant chunk should give NDCG=0, got {ndcg_worst}"

    # Empty selection → 0.0
    assert calculate_ndcg([], chunk_scores) == 0.0

    # No relevant items at all
    assert calculate_ndcg([0], {0: 0, 1: 0}) == 0.0

    # Single perfect selection
    ndcg_single = calculate_ndcg([1], chunk_scores)
    assert ndcg_single == 1.0, f"Selecting the single best chunk should give NDCG=1.0, got {ndcg_single}"


def test_embedding_scorer_separation():
    # Verify that trigram-based EmbeddingScorer yields better separation than CharFrequencyScorer
    question = "Agreement governing Delaware law"
    chunks = [
        "This agreement is governed by the state of Delaware law.", # relevant
        "The cat likes to sleep under the warm yellow sun." # completely irrelevant
    ]
    
    old_scorer = CharFrequencyScorer()
    new_scorer = EmbeddingScorer()
    
    old_scores = old_scorer.score(question, chunks)
    new_scores = new_scorer.score(question, chunks)
    
    # Old separation (relevant - irrelevant score)
    old_sep = old_scores[0] - old_scores[1]
    # New separation
    new_sep = new_scores[0] - new_scores[1]
    
    print(f"Old separation: {old_sep:.4f}, New separation: {new_sep:.4f}")
    assert new_sep > old_sep, f"New separation ({new_sep}) should be greater than old separation ({old_sep})"
