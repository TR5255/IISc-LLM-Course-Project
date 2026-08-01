"""
reporting/exporters.py
----------------------
Export utilities for saving comparative benchmark metric tables to CSV and JSON formats.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List


def export_to_csv(results: List[Dict[str, Any]], filepath: str) -> str:
    """Exports benchmark results list to a clean CSV table."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if not results:
        return filepath

    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return filepath


def export_to_json(results: List[Dict[str, Any]], filepath: str) -> str:
    """Exports benchmark results to structured JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return filepath
