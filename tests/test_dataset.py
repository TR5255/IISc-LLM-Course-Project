from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from models.scorer import RandomScorer
from baselines.bm25 import BM25Scorer
from baselines.tfidf import TFIDFScorer
from baselines.embedding import EmbeddingScorer, CharFrequencyScorer
from models.policies import TopKPolicy, ThresholdPolicy
from models.router import SmartAIRouter
from evaluation.retention import calculate_precision_recall, calculate_f1, calculate_ndcg
from evaluation.compression import calculate_token_compression_ratio

def test_benchmark_data_validation():
    # Load dataset
    loader = BenchmarkDatasetLoader(json_path="data/datasets/raw/benchmark_data.json")
    dataset = loader.load()
    
    assert len(dataset) >= 10, "Benchmark must contain at least 10 entries."
    
    for idx, item in enumerate(dataset.items):
        item_id = item.document_id
        
        # Verify item document meta
        assert item_id, f"Item at index {idx} must have non-empty document_id."
        assert item.document_text, f"Item '{item_id}' must have non-empty document_text."
        assert item.question, f"Item '{item_id}' must have non-empty question."
        assert item.answer, f"Item '{item_id}' must have non-empty answer."
        
        # Verify difficulty
        assert item.difficulty in ["easy", "medium", "hard"], f"Item '{item_id}' has invalid difficulty: {item.difficulty}"
        
        # Verify provenance
        prov = item.provenance
        assert prov, f"Item '{item_id}' must specify provenance."
        assert prov.get("source_type") in ["public", "synthetic"], f"Item '{item_id}' provenance source_type must be public/synthetic."
        assert prov.get("source_name"), f"Item '{item_id}' provenance source_name must not be empty."
        assert prov.get("license"), f"Item '{item_id}' provenance license must not be empty."
        
        # Verify chunks
        assert item.chunks, f"Item '{item_id}' must contain a list of chunks."
        
        # Verify chunk IDs uniqueness and ordering
        chunk_ids = [c.id for c in item.chunks]
        assert chunk_ids == list(range(len(item.chunks))), f"Item '{item_id}' chunks must have sequential 0-indexed IDs."
        
        has_relevant = False
        for c in item.chunks:
            # Check metadata fields
            assert c.text, f"Item '{item_id}', chunk '{c.id}' text must not be empty."
            assert c.token_count > 0, f"Item '{item_id}', chunk '{c.id}' token count must be greater than zero."
            assert isinstance(c.is_relevant, bool), f"Item '{item_id}', chunk '{c.id}' is_relevant must be a boolean."
            assert c.relevance_score in [0, 1, 2, 3], f"Item '{item_id}', chunk '{c.id}' relevance score must be within range 0 to 3."
            
            # Check binary relevance alignment with graded scores
            if c.is_relevant:
                has_relevant = True
                assert c.relevance_score > 0, f"Item '{item_id}', chunk '{c.id}' marked relevant but score is 0."
            else:
                assert c.relevance_score == 0, f"Item '{item_id}', chunk '{c.id}' marked irrelevant but score is non-zero."
                
            # Verify positions alignment if present
            if c.start_position is not None and c.end_position is not None:
                assert c.start_position < c.end_position, f"Item '{item_id}', chunk '{c.id}' character spans are invalid."
                # Extract text using spans to verify alignment
                extracted = item.document_text[c.start_position:c.end_position]
                # Allow minor padding checks, but strip to compare
                assert extracted.strip() == c.text.strip(), f"Item '{item_id}', chunk '{c.id}' position slice text does not match chunk text."

        assert has_relevant, f"Item '{item_id}' must have at least one marked relevant chunk."


def test_random_baseline_execution():
    loader = BenchmarkDatasetLoader(json_path="data/datasets/raw/benchmark_data.json")
    dataset = loader.load()
    item = dataset.items[0]
    
    scorer = RandomScorer(seed=42)
    policy = TopKPolicy(k=1)
    router = SmartAIRouter(scorer=scorer, policy=policy)
    
    chunks_text = [c.text for c in item.chunks]
    selected = router.route(item.question, chunks_text)
    
    assert len(selected) == 1, "Random baseline with TopK(1) must yield exactly one chunk."
    assert selected[0] in chunks_text, "Selected chunk must be in original chunk text list."


def test_all_scorers_benchmark_run():
    """Verifies all 4 scorers produce valid metric outputs across the full benchmark."""
    loader = BenchmarkDatasetLoader(json_path="data/datasets/raw/benchmark_data.json")
    dataset = loader.load()

    all_scorers = [
        ("random", RandomScorer(seed=42)),
        ("bm25", BM25Scorer()),
        ("tfidf", TFIDFScorer()),
        ("char_frequency", CharFrequencyScorer()),
        ("embedding", EmbeddingScorer()),
    ]
    policy = ThresholdPolicy(threshold=0.4)

    for scorer_name, scorer in all_scorers:
        router = SmartAIRouter(scorer=scorer, policy=policy)

        for item in dataset.items:
            chunks_text = [c.text for c in item.chunks]
            selected = router.route(item.question, chunks_text)

            # Selected chunks must be a subset of the original
            for s in selected:
                assert s in chunks_text, f"{scorer_name}: selected chunk not in original for {item.document_id}"

            # Map to IDs and compute metrics
            selected_ids = []
            for sel_t in selected:
                for c in item.chunks:
                    if c.text == sel_t and c.id not in selected_ids:
                        selected_ids.append(c.id)
                        break

            gold_ids = {c.id for c in item.chunks if c.is_relevant}
            precision, recall = calculate_precision_recall(selected_ids, gold_ids)
            f1 = calculate_f1(precision, recall)
            chunk_rel_map = {c.id: c.relevance_score for c in item.chunks}
            ndcg = calculate_ndcg(selected_ids, chunk_rel_map)
            comp = calculate_token_compression_ratio(chunks_text, selected)

            # All metrics must be valid floats in [0, 1]
            for metric_name, val in [("precision", precision), ("recall", recall),
                                     ("f1", f1), ("ndcg", ndcg), ("comp", comp)]:
                assert 0.0 <= val <= 1.0, (
                    f"{scorer_name}/{item.document_id}: {metric_name}={val} out of [0,1]"
                )

print("test_dataset.py tests parsed correctly.")
