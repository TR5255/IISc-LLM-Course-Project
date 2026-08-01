"""
ui/backend/app.py
------------------
FastAPI REST API server for Smart AI Router framework.
Provides endpoints for:
  - Benchmark All (Dynamic discovery & execution of all routing principles)
  - Interactive Research Report Viewer & ZIP Package Downloads
  - Side-by-side Experiment Run Comparisons
  - Historical Report Summary Retrieval
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from evaluation.gemini_provider import GeminiFlashProvider
from evaluation.llm_eval import DownstreamLLMEvaluator
from evaluation.retention import calculate_precision_recall, calculate_f1, calculate_ndcg
from evaluation.compression import calculate_compression_ratio, calculate_token_compression_ratio
from models.policies import TopKPolicy
from models.router import SmartAIRouter
from baselines import get_all_registered_routers, get_router_metadata
from storage.experiment_registry import ExperimentRegistry, ExperimentRun
from storage.artifact_store import ArtifactStore
from reporting.report_generator import PaperReportGenerator
from utils.logging import setup_logger

logger = setup_logger("api")

app = FastAPI(
    title="Smart AI Router Research Workbench API",
    description="API for Dynamic Context Routing Benchmarks and Paper-Ready Report Generation",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ExperimentRegistry()
artifact_store = ArtifactStore()
report_generator = PaperReportGenerator()
BENCHMARK_PATH = "data/datasets/raw/benchmark_data.json"


class BenchmarkAllRequest(BaseModel):
    """Request body for /api/benchmark-all — api_key is sent in body, not query string."""
    api_key: Optional[str] = None
    sample_size: Optional[int] = None  # None = use full dataset


@app.get("/api/status")
def get_status():
    """Returns platform condition, registry statistics, and per-principle metadata."""
    routers = get_all_registered_routers()
    meta = get_router_metadata()
    runs = registry.list_runs(limit=10)
    return {
        "status": "online",
        "dataset": "LexGLUE Legal QA Benchmark",
        "downstream_llm": "Gemini Flash (Fixed)",
        "registered_principles": [
            {
                "name": name,
                "requires_training": meta.get(name, {}).get("requires_training", False),
                "is_available": name in routers,
            }
            for name in meta
        ],
        "total_experiments": len(runs),
        "recent_runs": [r.run_id for r in runs[:5]],
    }


@app.post("/api/benchmark-all")
def run_benchmark_all(body: BenchmarkAllRequest = BenchmarkAllRequest()):
    """
    PRIMARY WORKFLOW:
    Dynamically discovers and executes EVERY registered routing principle.
    Uses real evaluation metrics from evaluation/retention.py and evaluation/compression.py.
    Per-principle errors are isolated — a failure in one principle does not abort the whole run.
    """
    routers = get_all_registered_routers()
    results: List[Dict[str, Any]] = []
    per_item_details: List[Dict[str, Any]] = []

    loader = BenchmarkDatasetLoader(BENCHMARK_PATH)
    items = loader.load()

    # Use full dataset unless caller specifies a sample_size
    if body.sample_size and body.sample_size < len(items):
        eval_items = items[: body.sample_size]
    else:
        eval_items = items

    provider = GeminiFlashProvider(api_key=body.api_key)
    evaluator = DownstreamLLMEvaluator(provider=provider)

    run_timestamp = int(time.time())
    run_id = f"benchmark_suite_{run_timestamp}"

    for principle_name, scorer in routers.items():
        try:
            router = SmartAIRouter(scorer=scorer, policy=TopKPolicy(k=2))

            precisions, recalls, f1s, ndcgs = [], [], [], []
            char_compressions, token_compressions = [], []
            accuracies, latencies, costs = [], [], []
            item_details_for_principle: List[Dict] = []

            for item in eval_items:
                text_chunks = [c.text for c in item.chunks]
                routed_chunks = router.route(question=item.question, chunks=text_chunks)
                compressed_text = "\n".join(routed_chunks)

                # --- Real retrieval metrics from evaluation/retention.py ---
                chunk_id_by_text = {c.text: c.id for c in item.chunks}
                selected_chunk_ids = [
                    chunk_id_by_text[t] for t in routed_chunks if t in chunk_id_by_text
                ]
                gold_relevant_ids = {c.id for c in item.chunks if c.is_relevant}
                chunk_relevance_scores = {c.id: c.relevance_score for c in item.chunks}

                precision, recall = calculate_precision_recall(selected_chunk_ids, gold_relevant_ids)
                f1 = calculate_f1(precision, recall)
                ndcg = calculate_ndcg(selected_chunk_ids, chunk_relevance_scores, k=3)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                ndcgs.append(ndcg)

                # --- Real compression metrics from evaluation/compression.py ---
                char_ratio = calculate_compression_ratio(item.document_text, compressed_text)
                token_ratio = calculate_token_compression_ratio(text_chunks, routed_chunks)
                char_compressions.append(round((1 - char_ratio) * 100, 2))
                token_compressions.append(round((1 - token_ratio) * 100, 2))

                # --- Downstream LLM evaluation ---
                ev = evaluator.evaluate_answer_possibility(
                    question=item.question,
                    compressed_text=compressed_text,
                    reference_answer=item.answer,
                    original_text=item.document_text,
                )
                accuracies.append(ev["accuracy_score"])
                latencies.append(ev["latency_sec"])
                costs.append(ev["estimated_cost_usd"])

                item_details_for_principle.append({
                    "document_id": item.document_id,
                    "principle": principle_name,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "ndcg@3": round(ndcg, 4),
                    "compression_pct": round((1 - char_ratio) * 100, 2),
                    "token_savings_pct": round((1 - token_ratio) * 100, 2),
                    "downstream_accuracy": round(ev["accuracy_score"], 4),
                    "selected_chunks": len(routed_chunks),
                    "total_chunks": len(text_chunks),
                })

            per_item_details.extend(item_details_for_principle)
            n = max(1, len(eval_items))

            res = {
                "routing_principle": principle_name,
                "precision": round(sum(precisions) / n, 4),
                "recall": round(sum(recalls) / n, 4),
                "f1": round(sum(f1s) / n, 4),
                "ndcg@3": round(sum(ndcgs) / n, 4),
                "compression_pct": round(sum(char_compressions) / n, 2),
                "token_savings_pct": round(sum(token_compressions) / n, 2),
                "avg_latency_sec": round(sum(latencies) / n, 4),
                "total_cost_usd": round(sum(costs), 6),
                "downstream_accuracy": round(sum(accuracies) / n, 4),
            }
            results.append(res)

            # Log to registry
            run_record = ExperimentRun(
                run_id=f"{run_id}_{principle_name}",
                timestamp=time.time(),
                routing_principle=principle_name,
                dataset_name="LexGLUE",
                downstream_model="Gemini Flash",
                num_samples=len(eval_items),
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
                config_json=json.dumps({"policy": "TopKPolicy", "k": 2}),
                metrics_json=json.dumps(res),
            )
            registry.log_run(run_record)
            logger.info(f"Benchmark completed for '{principle_name}': F1={res['f1']}, NDCG@3={res['ndcg@3']}")

        except Exception as e:
            logger.error(f"Benchmark FAILED for '{principle_name}': {e}")
            results.append({
                "routing_principle": principle_name,
                "error": str(e),
                "precision": None, "recall": None, "f1": None, "ndcg@3": None,
                "compression_pct": None, "token_savings_pct": None,
                "avg_latency_sec": None, "total_cost_usd": None,
                "downstream_accuracy": None,
            })

    # Generate full Paper Research Package
    # Filter out errored results for report generation
    valid_results = [r for r in results if r.get("error") is None]
    report_artifacts = report_generator.generate_report(results=valid_results, run_id=run_id)
    stats = report_generator.calculate_statistics(valid_results)

    # Persist per-item detail alongside the report
    detail_path = os.path.join("data/reports", run_id, "per_item_detail.json")
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    with open(detail_path, "w") as f:
        json.dump(per_item_details, f, indent=2)

    return {
        "message": f"Benchmark All completed successfully across {len(results)} principles.",
        "run_id": run_id,
        "summary_results": results,
        "statistics": stats,
        "report_artifacts": report_artifacts,
    }


@app.get("/api/reports")
def list_reports():
    """Lists generated report directories."""
    reports_dir = "data/reports"
    if not os.path.exists(reports_dir):
        return {"reports": []}

    runs = []
    for item in os.listdir(reports_dir):
        full_p = os.path.join(reports_dir, item)
        if os.path.isdir(full_p):
            runs.append({
                "run_id": item,
                "created_at": os.path.getctime(full_p),
                "has_pdf": os.path.exists(os.path.join(full_p, "research_report.pdf")),
                "has_zip": os.path.exists(os.path.join(reports_dir, f"{item}_package.zip")),
            })
    return {"reports": sorted(runs, key=lambda x: x["created_at"], reverse=True)}


@app.get("/api/reports/{run_id}/download-zip")
def download_report_zip(run_id: str):
    """Downloads full ZIP research package for a specific run."""
    zip_path = os.path.join("data/reports", f"{run_id}_package.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP package not found")
    return FileResponse(zip_path, filename=f"{run_id}_package.zip", media_type="application/zip")


@app.get("/api/reports/{run_id}/markdown")
def get_report_markdown(run_id: str):
    """Returns raw markdown content of a report."""
    md_path = os.path.join("data/reports", run_id, "research_report.md")
    if not os.path.exists(md_path):
        raise HTTPException(status_code=404, detail="Report markdown not found")
    with open(md_path, "r") as f:
        return {"content": f.read()}


@app.get("/api/reports/{run_id}/summary")
def get_report_summary(run_id: str):
    """Returns parsed summary_results + statistics for a historical run (for dashboard re-rendering)."""
    json_path = os.path.join("data/reports", run_id, "metrics_summary.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Metrics summary not found")
    with open(json_path, "r") as f:
        raw_results = json.load(f)
    stats = report_generator.calculate_statistics(raw_results)
    return {"summary_results": raw_results, "statistics": stats}
