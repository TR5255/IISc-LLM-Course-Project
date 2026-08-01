"""
reporting/narrative.py
-----------------------
Data-driven narrative generator for research reports and dashboards.
Avoids hardcoded prose by analyzing actual metrics to produce observations,
Pareto efficiency suggestions, and structured conclusions.
"""
from __future__ import annotations

from typing import Any, Dict, List
import math


def get_pareto_efficient_front(results: List[Dict[str, Any]]) -> List[str]:
    """
    Identifies Pareto-efficient routing principles maximizing both context
    compression (compression_pct) and downstream accuracy (downstream_accuracy).
    """
    valid = [r for r in results if r.get("compression_pct") is not None and r.get("downstream_accuracy") is not None]
    if not valid:
        return []

    efficient_front = []
    for candidate in valid:
        c_comp = candidate["compression_pct"]
        c_acc = candidate["downstream_accuracy"]
        dominated = False
        
        for other in valid:
            if other == candidate:
                continue
            o_comp = other["compression_pct"]
            o_acc = other["downstream_accuracy"]
            
            # other dominates candidate if other is at least as good in all objectives
            # and strictly better in at least one objective
            if (o_comp >= c_comp and o_acc >= c_acc) and (o_comp > c_comp or o_acc > c_acc):
                dominated = True
                break
                
        if not dominated:
            efficient_front.append(candidate["routing_principle"])
            
    return efficient_front


def generate_observations(results: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[str]:
    """
    Generates a list of factual data-driven observations.
    """
    observations = []
    valid_results = [r for r in results if r.get("error") is None]
    if not valid_results:
        return ["No valid benchmark data available to make observations."]

    # 1. Compare best F1 vs best accuracy
    bp = stats.get("best_performers", {})
    best_f1_principle = bp.get("highest_f1")
    best_acc_principle = bp.get("highest_accuracy")

    if best_f1_principle and best_acc_principle:
        if best_f1_principle == best_acc_principle:
            observations.append(
                f"The `{best_f1_principle}` strategy achieved alignment as both the highest "
                f"retrieval F1 score and the highest downstream reasoning accuracy."
            )
        else:
            observations.append(
                f"A trade-off was observed: `{best_f1_principle}` achieved the highest retrieval F1 score, "
                f"whereas downstream accuracy peaked under the `{best_acc_principle}` strategy."
            )

    # 2. Monotonic trend / correlation check between compression and accuracy
    comps = [r["compression_pct"] for r in valid_results if r.get("compression_pct") is not None]
    accs = [r["downstream_accuracy"] for r in valid_results if r.get("downstream_accuracy") is not None]
    
    if len(comps) >= 3 and len(accs) == len(comps):
        # Pearson correlation coefficient calculation
        mean_comp = sum(comps) / len(comps)
        mean_acc = sum(accs) / len(accs)
        
        num = sum((c - mean_comp) * (a - mean_acc) for c, a in zip(comps, accs))
        den_c = sum((c - mean_comp) ** 2 for c in comps)
        den_a = sum((a - mean_acc) ** 2 for a in accs)
        
        if den_c > 0 and den_a > 0:
            r_val = num / math.sqrt(den_c * den_a)
            if r_val < -0.3:
                observations.append(
                    f"There is a negative correlation (r = {r_val:.2f}) between context compression "
                    f"and reasoning accuracy, indicating that higher compression rates tend to prune "
                    f"information required for downstream correctness."
                )
            elif r_val > 0.3:
                observations.append(
                    f"A positive correlation (r = {r_val:.2f}) between compression and accuracy "
                    f"suggests that aggressive routing acted as an effective noise filter for the LLM."
                )
            else:
                observations.append(
                    f"The correlation between compression rates and downstream accuracy is weak (r = {r_val:.2f}), "
                    f"suggesting factors other than absolute context length drive LLM reasoning success."
                )

    # 3. Flag untrained neural_router if present and near random performance
    random_res = next((r for r in valid_results if r["routing_principle"] == "random"), None)
    neural_res = next((r for r in valid_results if r["routing_principle"] == "neural_router"), None)
    
    if random_res and neural_res:
        random_f1 = random_res.get("f1", 0.0) or 0.0
        neural_f1 = neural_res.get("f1", 0.0) or 0.0
        # If neural router is near random baseline (within 0.08 F1 score margin)
        if abs(neural_f1 - random_f1) <= 0.08:
            observations.append(
                "Notice: The `neural_router` performed close to the `random` baseline, "
                "indicating that the model is either untrained or executing in its mock fallback state."
            )

    # 4. Highlight best latency vs cost
    best_lat = bp.get("lowest_latency")
    best_cost = bp.get("lowest_cost")
    if best_lat and best_cost:
        if best_lat == best_cost:
            observations.append(
                f"`{best_lat}` is the most efficient principle, minimizing both latency and token cost."
            )
        else:
            observations.append(
                f"Latency minimization is best achieved via `{best_lat}`, while token cost is minimized via `{best_cost}`."
            )

    return observations


def generate_conclusions(results: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    """
    Generates a structured conclusions paragraph based on Pareto frontier.
    """
    valid_results = [r for r in results if r.get("error") is None]
    if not valid_results:
        return "No valid results available to formulate conclusions."

    pareto_candidates = get_pareto_efficient_front(valid_results)
    bp = stats.get("best_performers", {})
    
    highest_acc = bp.get("highest_accuracy", "N/A")
    lowest_cost = bp.get("lowest_cost", "N/A")
    
    pareto_str = ", ".join(f"`{p.upper()}`" for p in pareto_candidates)
    
    conclusion = (
        f"Based on multiobjective optimization of context compression and answer accuracy, "
        f"the Pareto-optimal routing strategies are: {pareto_str}. "
    )
    if highest_acc == lowest_cost:
        conclusion += f"The `{highest_acc.upper()}` strategy is dominant, maximizing both accuracy and cost savings."
    else:
        conclusion += (
            f"For deployments prioritizing absolute reasoning accuracy, `{highest_acc.upper()}` is recommended, "
            f"whereas for cost-sensitive environments, `{lowest_cost.upper()}` offers the most economic trade-off."
        )
        
    return conclusion
