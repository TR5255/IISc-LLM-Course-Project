"""
storage/experiment_registry.py
------------------------------
SQLite-backed Experiment Registry for persistent benchmark logging, metrics history,
and run comparisons.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRun:
    run_id: str
    timestamp: float
    routing_principle: str
    dataset_name: str
    downstream_model: str
    num_samples: int
    precision: float
    recall: float
    f1: float
    ndcg_3: float
    ndcg_5: float
    compression_pct: float
    token_savings_pct: float
    avg_latency_sec: float
    total_cost_usd: float
    downstream_accuracy: float
    config_json: str
    metrics_json: str


class ExperimentRegistry:
    """
    SQLite persistent storage for benchmark experiment runs.
    """

    def __init__(self, db_path: str = "data/registry.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    routing_principle TEXT,
                    dataset_name TEXT,
                    downstream_model TEXT,
                    num_samples INTEGER,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    ndcg_3 REAL,
                    ndcg_5 REAL,
                    compression_pct REAL,
                    token_savings_pct REAL,
                    avg_latency_sec REAL,
                    total_cost_usd REAL,
                    downstream_accuracy REAL,
                    config_json TEXT,
                    metrics_json TEXT
                )
            """)
            conn.commit()

    def log_run(self, run: ExperimentRun) -> None:
        """Insert or replace an experiment run record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO experiment_runs (
                    run_id, timestamp, routing_principle, dataset_name, downstream_model,
                    num_samples, precision, recall, f1, ndcg_3, ndcg_5, compression_pct,
                    token_savings_pct, avg_latency_sec, total_cost_usd, downstream_accuracy,
                    config_json, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id, run.timestamp, run.routing_principle, run.dataset_name,
                run.downstream_model, run.num_samples, run.precision, run.recall, run.f1,
                run.ndcg_3, run.ndcg_5, run.compression_pct, run.token_savings_pct,
                run.avg_latency_sec, run.total_cost_usd, run.downstream_accuracy,
                run.config_json, run.metrics_json
            ))
            conn.commit()

    def get_all_runs(self) -> List[ExperimentRun]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [self._row_to_run(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_run(row)
        return None

    def list_runs(self, limit: int = 50) -> List[ExperimentRun]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_runs ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [self._row_to_run(r) for r in rows]

    def _row_to_run(self, r: tuple) -> ExperimentRun:
        return ExperimentRun(
            run_id=r[0], timestamp=r[1], routing_principle=r[2], dataset_name=r[3],
            downstream_model=r[4], num_samples=r[5], precision=r[6], recall=r[7],
            f1=r[8], ndcg_3=r[9], ndcg_5=r[10], compression_pct=r[11],
            token_savings_pct=r[12], avg_latency_sec=r[13], total_cost_usd=r[14],
            downstream_accuracy=r[15], config_json=r[16], metrics_json=r[17]
        )
