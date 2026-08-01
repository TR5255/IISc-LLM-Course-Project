"""
tests/test_integration.py
--------------------------
End-to-end integration test verifying the complete pipeline:
Ingesting dataset -> routing chunks -> downstream evaluating ->
logging runs to SQLite -> generating research reports and zip packages.
"""
import os
import shutil
import pytest
import sqlite3
import json

from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from baselines import get_all_registered_routers
from models.policies import TopKPolicy
from models.router import SmartAIRouter
from evaluation.gemini_provider import GeminiFlashProvider
from evaluation.llm_eval import DownstreamLLMEvaluator
from storage.experiment_registry import ExperimentRegistry
from reporting.report_generator import PaperReportGenerator


def test_end_to_end_pipeline(tmp_path):
    # 1. Setup temporary workspace
    dataset_path = "data/datasets/raw/benchmark_data.json"
    assert os.path.exists(dataset_path), "Raw benchmark dataset missing."

    db_path = str(tmp_path / "test_registry.db")
    reports_dir = str(tmp_path / "test_reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Instantiate registry with custom db path
    registry = ExperimentRegistry(db_path=db_path)
    assert registry.db_path == db_path

    # Instantiate generator with custom reports output path
    generator = PaperReportGenerator(output_dir=reports_dir)

    # 2. Ingest dataset
    loader = BenchmarkDatasetLoader(dataset_path)
    items = loader.load()
    assert len(items) > 0
    # Evaluate on a slice of 2 items to run quickly
    eval_items = items[:2]

    # 3. Discover routers & evaluate
    routers = get_all_registered_routers()
    assert len(routers) >= 3  # random, bm25, tfidf, embedding, neural_router

    provider = GeminiFlashProvider(api_key="mock_key")  # executes mocked evaluation
    evaluator = DownstreamLLMEvaluator(provider=provider)

    summary_results = []
    
    for name, scorer in routers.items():
        router = SmartAIRouter(scorer=scorer, policy=TopKPolicy(k=2))
        
        precisions, recalls, f1s = [], [], []
        char_compressions, token_compressions = [], []
        accuracies, latencies, costs = [], [], []

        for item in eval_items:
            text_chunks = [c.text for c in item.chunks]
            routed_chunks = router.route(question=item.question, chunks=text_chunks)
            compressed_text = "\n".join(routed_chunks)

            # Retrieve scores from mock
            from evaluation.retention import calculate_precision_recall, calculate_f1, calculate_ndcg
            from evaluation.compression import calculate_compression_ratio, calculate_token_compression_ratio

            chunk_id_by_text = {c.text: c.id for c in item.chunks}
            selected_chunk_ids = [chunk_id_by_text[t] for t in routed_chunks if t in chunk_id_by_text]
            gold_relevant_ids = {c.id for c in item.chunks if c.is_relevant}
            chunk_relevance_scores = {c.id: c.relevance_score for c in item.chunks}

            precision, recall = calculate_precision_recall(selected_chunk_ids, gold_relevant_ids)
            f1 = calculate_f1(precision, recall)
            ndcg = calculate_ndcg(selected_chunk_ids, chunk_relevance_scores, k=3)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            char_ratio = calculate_compression_ratio(item.document_text, compressed_text)
            token_ratio = calculate_token_compression_ratio(text_chunks, routed_chunks)
            char_compressions.append((1 - char_ratio) * 100)
            token_compressions.append((1 - token_ratio) * 100)

            ev = evaluator.evaluate_answer_possibility(
                question=item.question,
                compressed_text=compressed_text,
                reference_answer=item.answer,
                original_text=item.document_text
            )
            accuracies.append(ev["accuracy_score"])
            latencies.append(ev["latency_sec"])
            costs.append(ev["estimated_cost_usd"])

        n = len(eval_items)
        res = {
            "routing_principle": name,
            "precision": sum(precisions) / n,
            "recall": sum(recalls) / n,
            "f1": sum(f1s) / n,
            "ndcg@3": sum(f1s) / n,  # mock ndcg representation
            "compression_pct": sum(char_compressions) / n,
            "token_savings_pct": sum(token_compressions) / n,
            "avg_latency_sec": sum(latencies) / n,
            "total_cost_usd": sum(costs),
            "downstream_accuracy": sum(accuracies) / n
        }
        summary_results.append(res)

        # 4. Assert SQLite Experiment registry logging
        from storage.experiment_registry import ExperimentRun
        run_record = ExperimentRun(
            run_id=f"test_integration_{name}",
            timestamp=1785500000.0,
            routing_principle=name,
            dataset_name="LexGLUE",
            downstream_model="Gemini Flash",
            num_samples=n,
            precision=res["precision"],
            recall=res["recall"],
            f1=res["f1"],
            ndcg_3=res["ndcg@3"],
            ndcg_5=res["ndcg@3"],
            compression_pct=res["compression_pct"],
            token_savings_pct=res["token_savings_pct"],
            avg_latency_sec=res["avg_latency_sec"],
            total_cost_usd=res["total_cost_usd"],
            downstream_accuracy=res["downstream_accuracy"],
            config_json="{}",
            metrics_json="{}",
        )
        registry.log_run(run_record)

    # Query registry to verify persistence
    logged_runs = registry.list_runs()
    assert len(logged_runs) == len(routers)
    assert any(r.routing_principle == "bm25" for r in logged_runs)

    # 5. Generate Multi-Format Research Report Package
    run_id = "test_run_e2e"
    outputs = generator.generate_report(results=summary_results, run_id=run_id)

    assert os.path.exists(outputs["markdown"])
    assert os.path.exists(outputs["csv"])
    assert os.path.exists(outputs["json"])
    assert os.path.exists(outputs["zip_package"])

    # Verify report sections are generated and present
    with open(outputs["markdown"], "r", encoding="utf-8") as f:
        md = f.read()
        assert "## 1. Executive Summary & Key Findings" in md
        assert "## 2. Methodology & Experimental Setup" in md
        assert "## 3. Comparative Benchmark Matrix" in md
        assert "## 4. Statistical Summary" in md
        assert "## 5. Quantitative Observations" in md
        assert "## 6. Strategic Recommendations & Conclusions" in md
        assert "## 7. Visualizations & Figures" in md
        assert "Figure 1:" in md
        assert "Figure 5:" in md
        assert "Figure 6:" in md
