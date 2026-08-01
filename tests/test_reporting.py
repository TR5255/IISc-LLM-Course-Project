"""
tests/test_reporting.py
-----------------------
Unit tests verifying reporting generation, exporters, and plot output functionality.
"""
import os
import shutil
import pytest

from reporting.report_generator import PaperReportGenerator
from reporting.exporters import export_to_csv, export_to_json
from reporting.plots import generate_benchmark_plots


@pytest.fixture
def sample_results():
    return [
        {
            "routing_principle": "bm25",
            "precision": 0.30,
            "recall": 0.25,
            "f1": 0.2727,
            "ndcg@3": 0.33,
            "compression_pct": 18.1,
            "token_savings_pct": 20.0,
            "avg_latency_sec": 0.0144,
            "total_cost_usd": 0.00015,
            "downstream_accuracy": 0.3333,
        },
        {
            "routing_principle": "tfidf",
            "precision": 0.75,
            "recall": 0.70,
            "f1": 0.7241,
            "ndcg@3": 0.73,
            "compression_pct": 64.9,
            "token_savings_pct": 65.0,
            "avg_latency_sec": 0.0517,
            "total_cost_usd": 0.00008,
            "downstream_accuracy": 0.8333,
        },
        {
            "routing_principle": "embedding",
            "precision": 0.56,
            "recall": 1.00,
            "f1": 0.7179,
            "ndcg@3": 0.67,
            "compression_pct": 100.0,
            "token_savings_pct": 0.0,
            "avg_latency_sec": 0.1960,
            "total_cost_usd": 0.00030,
            "downstream_accuracy": 1.0000,
        },
    ]


def test_exporters(sample_results, tmp_path):
    csv_file = str(tmp_path / "test.csv")
    json_file = str(tmp_path / "test.json")

    export_to_csv(sample_results, csv_file)
    export_to_json(sample_results, json_file)

    assert os.path.exists(csv_file)
    assert os.path.exists(json_file)


def test_plots(sample_results, tmp_path):
    plots_dir = str(tmp_path / "figures")
    plots = generate_benchmark_plots(sample_results, output_dir=plots_dir)

    try:
        import matplotlib
        assert "quality_metrics" in plots
        assert "compression_vs_accuracy" in plots
        assert os.path.exists(plots["quality_metrics"])
        assert os.path.exists(plots["compression_vs_accuracy"])
    except ImportError:
        assert plots == {}


def test_paper_report_generator(sample_results, tmp_path):
    output_dir = str(tmp_path / "reports")
    generator = PaperReportGenerator(output_dir=output_dir)

    outputs = generator.generate_report(
        results=sample_results,
        run_id="test_run_123",
    )

    assert os.path.exists(outputs["markdown"])
    assert os.path.exists(outputs["csv"])
    assert os.path.exists(outputs["json"])
    assert os.path.exists(outputs["zip_package"])

    with open(outputs["markdown"], "r", encoding="utf-8") as fh:
        content = fh.read()
        assert "Executive Summary" in content
        assert "TFIDF" in content
        assert "LexGLUE" in content
