"""
storage/artifact_store.py
-------------------------
Artifact store for saving and loading run outputs, evaluation reports,
and model checkpoints.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class ArtifactStore:
    """
    Manages JSON/JSONL artifact persistence under data/artifacts/ or custom paths.
    """

    def __init__(self, base_dir: str = "data/artifacts"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_json(self, filename: str, data: Dict[str, Any]) -> str:
        filepath = os.path.join(self.base_dir, filename)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return filepath

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def list_artifacts(self) -> list[str]:
        if not os.path.exists(self.base_dir):
            return []
        return [f for f in os.listdir(self.base_dir) if f.endswith(".json") or f.endswith(".csv") or f.endswith(".pdf") or f.endswith(".md")]
