#!/usr/bin/env bash
# =============================================================================
# download_cuad.sh — Download the CUAD v1 dataset
# =============================================================================
# CUAD (Contract Understanding Atticus Dataset) is a collection of 510
# commercial legal contracts annotated with 41 clause-type labels.
#
# Source: https://huggingface.co/datasets/cuad
# License: CC BY 4.0
#
# This script downloads the raw dataset to data/external/cuad/raw/.
# The raw data is NOT committed to the repository (.gitignore excludes it).
# Run this script once locally before using cuad_loader.py.
# =============================================================================

set -euo pipefail

RAW_DIR="data/external/cuad/raw"
mkdir -p "$RAW_DIR"

echo "[CUAD] Downloading CUAD_v1.json from Hugging Face..."
wget --show-progress \
  -O "${RAW_DIR}/CUAD_v1.json" \
  "https://huggingface.co/datasets/cuad/resolve/main/CUAD_v1.json"

echo "[CUAD] Download complete: ${RAW_DIR}/CUAD_v1.json"
echo "[CUAD] You can now run: PYTHONPATH=. python -m data.external.cuad.cuad_loader"
