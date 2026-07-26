import argparse
import json
import os
import time
from typing import Dict, Any, List, Optional

from utils.config import load_config
from utils.logging import setup_logger
from utils.tracker import ExperimentTracker

from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from models.scorer import RandomScorer, BaseScorer
from baselines.bm25 import BM25Scorer
from baselines.tfidf import TFIDFScorer
from baselines.embedding import EmbeddingScorer, CharFrequencyScorer

from models.policies import ThresholdPolicy, TopKPolicy, BasePolicy
from models.router import SmartAIRouter

from evaluation.compression import calculate_token_compression_ratio
from evaluation.retention import calculate_precision_recall, calculate_f1, calculate_ndcg
from evaluation.llm_eval import DownstreamLLMEvaluator


# ---------------------------------------------------------------------------
# Scorer factory
# ---------------------------------------------------------------------------

SCORER_REGISTRY = {
    "random": lambda cfg: RandomScorer(seed=getattr(cfg.model, "seed", 42)),
    "bm25":   lambda _: BM25Scorer(),
    "tfidf":  lambda _: TFIDFScorer(),
    "char_frequency": lambda _: CharFrequencyScorer(),
    "embedding": lambda _: EmbeddingScorer(),
}


def _make_scorer(name: str, config: Any) -> BaseScorer:
    key = name.lower()
    if key not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer: {name}")
    return SCORER_REGISTRY[key](config)


# ---------------------------------------------------------------------------
# Core evaluation (single scorer + single policy)
# ---------------------------------------------------------------------------

def run_evaluation(
    config: Any,
    scorer: BaseScorer,
    policy: BasePolicy,
    scorer_name: str,
    policy_label: str,
    logger: Any,
) -> Dict[str, Any]:
    """Runs a single scorer+policy evaluation across the benchmark dataset.

    Returns a dict with 'overall', per-difficulty, and 'per_item' detail records.
    """
    # Load dataset
    loader = BenchmarkDatasetLoader(json_path="data/datasets/raw/benchmark_data.json")
    dataset = loader.load()

    evaluator = DownstreamLLMEvaluator(
        downstream_model_name=getattr(config.evaluation, "downstream_model", "mock-gpt-4o")
    )

    router = SmartAIRouter(scorer=scorer, policy=policy)

    difficulty_stats: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}
    per_item_records: List[Dict[str, Any]] = []

    all_relevant_scores = []
    all_irrelevant_scores = []
    all_scores = []

    for item in dataset.items:
        chunks_text = [c.text for c in item.chunks]

        # Explicitly get the score distribution across all chunks
        scores = scorer.score(item.question, chunks_text)
        for idx, score in enumerate(scores):
            all_scores.append(score)
            if item.chunks[idx].is_relevant:
                all_relevant_scores.append(score)
            else:
                all_irrelevant_scores.append(score)

        start_time = time.perf_counter()
        selected_text_list = policy.select(chunks_text, scores)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Map selected texts → chunk IDs (preserving selection order for NDCG)
        selected_ids: List[int] = []
        for sel_t in selected_text_list:
            for c in item.chunks:
                if c.text == sel_t and c.id not in selected_ids:
                    selected_ids.append(c.id)
                    break

        gold_ids = {c.id for c in item.chunks if c.is_relevant}
        precision, recall = calculate_precision_recall(selected_ids, gold_ids)
        f1 = calculate_f1(precision, recall)

        # NDCG using graded relevance scores
        chunk_rel_map = {c.id: c.relevance_score for c in item.chunks}
        ndcg = calculate_ndcg(selected_ids, chunk_rel_map)

        comp_ratio = calculate_token_compression_ratio(chunks_text, selected_text_list)

        compressed_text = " ".join(selected_text_list)
        llm_res = evaluator.evaluate_answer_possibility(
            question=item.question,
            compressed_text=compressed_text,
            reference_answer=item.answer,
        )
        accuracy = llm_res["accuracy_score"]

        record = {
            "document_id": item.document_id,
            "difficulty": item.difficulty,
            "scorer": scorer_name,
            "policy": policy_label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
            "compression_ratio": comp_ratio,
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            "selected_chunk_ids": selected_ids,
            "gold_chunk_ids": list(gold_ids),
        }
        per_item_records.append(record)
        difficulty_stats[item.difficulty].append(record)

    # Aggregate helper
    def _avg(records: List[Dict], key: str) -> float:
        if not records:
            return 0.0
        return sum(r[key] for r in records) / len(records)

    mean_relevant = sum(all_relevant_scores) / len(all_relevant_scores) if all_relevant_scores else 0.0
    mean_irrelevant = sum(all_irrelevant_scores) / len(all_irrelevant_scores) if all_irrelevant_scores else 0.0
    min_similarity = min(all_scores) if all_scores else 0.0
    max_similarity = max(all_scores) if all_scores else 0.0
    similarity_separation = mean_relevant - mean_irrelevant

    overall = {
        "precision": _avg(per_item_records, "precision"),
        "recall": _avg(per_item_records, "recall"),
        "f1": _avg(per_item_records, "f1"),
        "ndcg": _avg(per_item_records, "ndcg"),
        "compression_ratio": _avg(per_item_records, "compression_ratio"),
        "accuracy": _avg(per_item_records, "accuracy"),
        "latency_ms": _avg(per_item_records, "latency_ms"),
        "mean_relevant_score": mean_relevant,
        "mean_irrelevant_score": mean_irrelevant,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "similarity_separation": similarity_separation,
    }

    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        recs = difficulty_stats[diff]
        by_difficulty[diff] = {
            "count": len(recs),
            "precision": _avg(recs, "precision"),
            "recall": _avg(recs, "recall"),
            "f1": _avg(recs, "f1"),
            "ndcg": _avg(recs, "ndcg"),
            "compression_ratio": _avg(recs, "compression_ratio"),
            "accuracy": _avg(recs, "accuracy"),
        }

    return {
        "scorer": scorer_name,
        "policy": policy_label,
        "overall": overall,
        "by_difficulty": by_difficulty,
        "per_item": per_item_records,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_HDR = (
    f"{'Scorer':<12} | {'Policy':<18} | {'Prec':<6} | {'Recall':<6} "
    f"| {'F1':<6} | {'NDCG':<6} | {'Comp%':<6} | {'Acc':<6} | {'Lat(ms)':<9}"
)
_SEP = "-" * len(_HDR)


def _fmt_row(r: Dict[str, Any]) -> str:
    o = r["overall"]
    return (
        f"{r['scorer']:<12} | {r['policy']:<18} | {o['precision']:<6.1%} | {o['recall']:<6.1%} "
        f"| {o['f1']:<6.1%} | {o['ndcg']:<6.2f} | {o['compression_ratio']:<6.1%} "
        f"| {o['accuracy']:<6.1%} | {o['latency_ms']:.4f}"
    )


def print_results_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * len(_HDR))
    print("BENCHMARK EXPERIMENT RESULTS")
    print("=" * len(_HDR))
    print(_HDR)
    print(_SEP)
    for r in results:
        print(_fmt_row(r))
    print("=" * len(_HDR))

    # Score distribution table
    dist_hdr = f"{'Scorer':<12} | {'Policy':<18} | {'Avg Rel':<7} | {'Avg Irrel':<9} | {'Min Sim':<8} | {'Max Sim':<8} | {'Sep':<5}"
    dist_sep = "-" * len(dist_hdr)
    print("\n" + "=" * len(dist_hdr))
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * len(dist_hdr))
    print(dist_hdr)
    print(dist_sep)
    for r in results:
        o = r["overall"]
        avg_rel = o.get("mean_relevant_score", 0.0)
        avg_irrel = o.get("mean_irrelevant_score", 0.0)
        min_sim = o.get("min_similarity", 0.0)
        max_sim = o.get("max_similarity", 0.0)
        sep = o.get("similarity_separation", 0.0)
        print(f"{r['scorer']:<12} | {r['policy']:<18} | {avg_rel:<7.3f} | {avg_irrel:<9.3f} | {min_sim:<8.3f} | {max_sim:<8.3f} | {sep:<5.3f}")
    print("=" * len(dist_hdr))

    # Difficulty breakdown
    print("\nDifficulty Breakdown:")
    for diff in ["easy", "medium", "hard"]:
        print(f"\n  {diff.upper()}")
        print(f"  {'Scorer':<12} | {'Policy':<18} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'NDCG':<6} | {'Acc':<6}")
        print("  " + "-" * 76)
        for r in results:
            d = r["by_difficulty"].get(diff, {})
            print(
                f"  {r['scorer']:<12} | {r['policy']:<18} | "
                f"{d.get('precision',0):<6.1%} | {d.get('recall',0):<6.1%} | "
                f"{d.get('f1',0):<6.1%} | {d.get('ndcg',0):<6.2f} | "
                f"{d.get('accuracy',0):<6.1%}"
            )
    print("=" * len(_HDR))


# ---------------------------------------------------------------------------
# JSON report export
# ---------------------------------------------------------------------------

def export_json_report(
    experiment_id: str,
    config: Any,
    results: List[Dict[str, Any]],
    runs_dir: str = "experiments/runs",
):
    """Writes a machine-readable JSON report with full reproducibility metadata."""
    os.makedirs(runs_dir, exist_ok=True)

    # Strip per-item detail of non-serialisable bits (all values are primitives already)
    report = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config.to_dict(),
        "scorers_evaluated": list({r["scorer"] for r in results}),
        "policies_evaluated": list({r["policy"] for r in results}),
        "results": results,
    }

    path = os.path.join(runs_dir, f"{experiment_id}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def export_flat_results(
    experiment_id: str,
    results: List[Dict[str, Any]],
    runs_dir: str = "experiments/runs",
) -> str:
    """Writes a flat list of experiment runs with standard keys for easy analysis."""
    os.makedirs(runs_dir, exist_ok=True)
    flat_records = []
    for r in results:
        policy_label = r["policy"]
        policy_type = "threshold" if "threshold" in policy_label else "top_k"
        param_str = policy_label.split("=")[-1]
        parameter = float(param_str) if policy_type == "threshold" else int(param_str)
        
        o = r["overall"]
        rec = {
            "scorer": r["scorer"],
            "policy": policy_type,
            "parameter": parameter,
            "precision": o["precision"],
            "recall": o["recall"],
            "f1": o["f1"],
            "ndcg": o["ndcg"],
            "compression_ratio": o["compression_ratio"],
            "token_retention": o["recall"],
            "downstream_accuracy": o["accuracy"],
        }
        flat_records.append(rec)
        
    path = os.path.join(runs_dir, f"{experiment_id}_flat.json")
    with open(path, "w") as f:
        json.dump(flat_records, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run Smart AI Router experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/base_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run a parameter sweep across thresholds and k values.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_single(config: Any, logger: Any) -> List[Dict[str, Any]]:
    """Run all scorers with the single policy specified in the config."""
    policy_type = config.policy.type.lower()
    if policy_type == "threshold":
        policy = ThresholdPolicy(threshold=getattr(config.policy, "threshold", 0.5))
        policy_label = f"threshold={getattr(config.policy, 'threshold', 0.5)}"
    elif policy_type == "top_k":
        k = getattr(config.policy, "k", 2)
        policy = TopKPolicy(k=k)
        policy_label = f"top_k={k}"
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    results = []
    for scorer_name in SCORER_REGISTRY:
        logger.info(f"Evaluating: {scorer_name} / {policy_label}")
        scorer = _make_scorer(scorer_name, config)
        res = run_evaluation(config, scorer, policy, scorer_name, policy_label, logger)
        results.append(res)

    return results


def run_sweep(config: Any, logger: Any) -> List[Dict[str, Any]]:
    """Sweep across threshold values and k values for every scorer."""
    sweep_cfg = getattr(config, "sweep", None)
    if sweep_cfg:
        thresholds = getattr(sweep_cfg, "thresholds", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
        k_values = getattr(sweep_cfg, "k_values", [1, 2, 3, 5, 10])
    else:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        k_values = [1, 2, 3, 5, 10]

    results = []
    for scorer_name in SCORER_REGISTRY:
        scorer = _make_scorer(scorer_name, config)

        for t in thresholds:
            policy = ThresholdPolicy(threshold=t)
            label = f"threshold={t}"
            logger.info(f"Sweep: {scorer_name} / {label}")
            res = run_evaluation(config, scorer, policy, scorer_name, label, logger)
            results.append(res)

        for k in k_values:
            policy = TopKPolicy(k=k)
            label = f"top_k={k}"
            logger.info(f"Sweep: {scorer_name} / {label}")
            res = run_evaluation(config, scorer, policy, scorer_name, label, logger)
            results.append(res)

    return results


def main():
    args = parse_args()
    logger = setup_logger("run_experiment")

    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    if args.sweep:
        logger.info("Starting parameter sweep...")
        results = run_sweep(config, logger)
    else:
        logger.info("Starting single-config evaluation...")
        results = run_single(config, logger)

    # Console output
    print_results_table(results)

    # Log to tracker
    tracker = ExperimentTracker()
    for r in results:
        o = r["overall"]
        metrics = {
            "precision": o["precision"],
            "recall": o["recall"],
            "f1": o["f1"],
            "ndcg": o["ndcg"],
            "compression_ratio": o["compression_ratio"],
            "retention_score": o["recall"],
            "downstream_accuracy": o["accuracy"],
            "latency_ms": o["latency_ms"],
        }
        mod_config = config.to_dict()
        mod_config["model"]["name"] = r["scorer"]
        mod_config["policy"]["label"] = r["policy"]
        tracker.log_run(f"{config.experiment.id}_{r['scorer']}_{r['policy']}", mod_config, metrics)

    # JSON report
    report_path = export_json_report(config.experiment.id, config, results)
    logger.info(f"JSON report saved to: {report_path}")

    # Flat JSON report for analysis
    flat_path = export_flat_results(config.experiment.id, results)
    logger.info(f"Flat JSON report saved to: {flat_path}")


if __name__ == "__main__":
    main()
