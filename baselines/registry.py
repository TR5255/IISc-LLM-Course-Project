"""
baselines/registry.py
---------------------
Centralized router registration mechanism for Smart AI Router.
Provides a decorator-based registration system and dynamic instantiation
with structured logging.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type
from utils.logging import setup_logger

logger = setup_logger("router_registry")

# --- Registry state ---
ROUTER_REGISTRY: Dict[str, Type] = {}
ROUTER_METADATA: Dict[str, Dict[str, Any]] = {}


def register_router(name: str, requires_training: bool = False) -> Callable:
    """
    Class decorator to register a BaseScorer subclass as a routing principle.

    Usage:
        @register_router("my_scorer")
        class MyScorer(BaseScorer):
            ...
    """
    def decorator(cls):
        ROUTER_REGISTRY[name] = cls
        ROUTER_METADATA[name] = {
            "requires_training": requires_training,
            "class": cls.__name__,
        }
        return cls
    return decorator


def get_all_registered_routers() -> Dict[str, Any]:
    """
    Instantiates and returns all dynamically registered routing principles.
    Routers that fail to instantiate (e.g., missing trained weights) are skipped
    with a structured log message rather than silently swallowed.
    """
    instances = {}
    for name, cls in ROUTER_REGISTRY.items():
        meta = ROUTER_METADATA.get(name, {})
        try:
            instances[name] = cls()
        except Exception as e:
            req_training = meta.get("requires_training", False)
            logger.warning(
                f"Skipped router '{name}' (class={cls.__name__}, "
                f"requires_training={req_training}): {e}"
            )
    return instances


def get_router_metadata() -> Dict[str, Dict[str, Any]]:
    """Returns metadata for all registered routers."""
    return dict(ROUTER_METADATA)
