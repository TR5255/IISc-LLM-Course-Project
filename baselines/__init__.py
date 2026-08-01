"""
baselines/__init__.py
---------------------
Dynamic router discovery and registry for Smart AI Router platform.
Automatically discovers and instantiates all available baseline, learned,
and neural routing principles.
"""
from typing import Dict, Type
from models.scorer import BaseScorer, RandomScorer
from baselines.bm25 import BM25Scorer
from baselines.tfidf import TFIDFScorer
from baselines.embedding import EmbeddingScorer

ROUTER_REGISTRY: Dict[str, Type[BaseScorer]] = {
    "random": RandomScorer,
    "bm25": BM25Scorer,
    "tfidf": TFIDFScorer,
    "embedding": EmbeddingScorer,
}

# Dynamic discovery for optional learned / neural models
try:
    from models.neural_router import NeuralRouterScorer
    ROUTER_REGISTRY["neural_router"] = NeuralRouterScorer
except ImportError:
    pass


def get_all_registered_routers() -> Dict[str, BaseScorer]:
    """Instantiates and returns all dynamically registered routing principles."""
    instances = {}
    for name, cls in ROUTER_REGISTRY.items():
        try:
            instances[name] = cls()
        except Exception:
            # If a model requires trained weights that don't exist yet, skip gracefully
            pass
    return instances
