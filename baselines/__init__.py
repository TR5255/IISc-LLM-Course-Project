"""
baselines/__init__.py
---------------------
Backward-compatible re-export of the router registry.
Populates ROUTER_REGISTRY with the four baselines at import time and
dynamically discovers optional learned/neural models.
"""
from baselines.registry import (
    ROUTER_REGISTRY,
    ROUTER_METADATA,
    register_router,
    get_all_registered_routers,
    get_router_metadata,
)
from models.scorer import BaseScorer, RandomScorer
from baselines.bm25 import BM25Scorer
from baselines.tfidf import TFIDFScorer
from baselines.embedding import EmbeddingScorer

# --- Baseline registrations (always available, require no training) ---
ROUTER_REGISTRY["random"] = RandomScorer
ROUTER_METADATA["random"] = {"requires_training": False, "class": "RandomScorer"}

ROUTER_REGISTRY["bm25"] = BM25Scorer
ROUTER_METADATA["bm25"] = {"requires_training": False, "class": "BM25Scorer"}

ROUTER_REGISTRY["tfidf"] = TFIDFScorer
ROUTER_METADATA["tfidf"] = {"requires_training": False, "class": "TFIDFScorer"}

ROUTER_REGISTRY["embedding"] = EmbeddingScorer
ROUTER_METADATA["embedding"] = {"requires_training": False, "class": "EmbeddingScorer"}

# --- Dynamic discovery: neural router (requires training, but participates
#     untrained with constant 0.5 baseline scores until trained weights exist) ---
try:
    from models.neural_router import NeuralRouter
    ROUTER_REGISTRY["neural_router"] = NeuralRouter
    ROUTER_METADATA["neural_router"] = {"requires_training": True, "class": "NeuralRouter"}
except ImportError:
    pass
