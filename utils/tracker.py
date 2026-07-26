import os
import csv
import json
import time
from typing import Any, Dict

class ExperimentTracker:
    """A lightweight tracker to log experiment parameters and metrics to CSV/JSON files."""
    def __init__(self, runs_dir: str = "experiments/runs"):
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)
        self.csv_path = os.path.join(runs_dir, "history.csv")
        self._init_csv()

    def _init_csv(self):
        # Initialize the CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            headers = [
                "experiment_id", "timestamp", "model_name", "dataset_version",
                "chunk_size", "policy_label", "threshold",
                "precision", "recall", "f1", "ndcg",
                "compression_ratio", "retention_score",
                "downstream_accuracy", "latency_ms"
            ]
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_run(self, experiment_id: str, config: Dict[str, Any], metrics: Dict[str, Any]):
        """Logs experiment run config and metrics to CSV and a single JSON run file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Flattened CSV log entry
        row = [
            experiment_id,
            timestamp,
            config.get("model", {}).get("name", "unknown"),
            config.get("dataset", {}).get("version", "unknown"),
            config.get("dataset", {}).get("chunk_size", -1),
            config.get("policy", {}).get("label", ""),
            config.get("policy", {}).get("threshold", -1.0),
            metrics.get("precision", -1.0),
            metrics.get("recall", -1.0),
            metrics.get("f1", -1.0),
            metrics.get("ndcg", -1.0),
            metrics.get("compression_ratio", -1.0),
            metrics.get("retention_score", -1.0),
            metrics.get("downstream_accuracy", -1.0),
            metrics.get("latency_ms", -1.0)
        ]
        
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        # JSON detailed dump
        run_data = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "config": config,
            "metrics": metrics
        }
        # Sanitise filename (replace = and . chars)
        safe_id = experiment_id.replace("=", "_").replace(".", "p")
        json_path = os.path.join(self.runs_dir, f"{safe_id}.json")
        with open(json_path, 'w') as f:
            json.dump(run_data, f, indent=4)
