import math
from typing import List, Set, Tuple, Optional

def calculate_precision_recall(
    selected_chunk_ids: List[int],
    gold_relevant_ids: Set[int]
) -> Tuple[float, float]:
    """Calculates precision and recall bounds for routed chunks.

    Args:
        selected_chunk_ids: List of integer IDs of chunks selected by the policy.
        gold_relevant_ids: Set of integer IDs of chunks annotated as is_relevant=True.

    Returns:
        Tuple[float, float]: (precision, recall)
    """
    if not selected_chunk_ids:
        precision = 0.0
        recall = 1.0 if not gold_relevant_ids else 0.0
        return precision, recall

    if not gold_relevant_ids:
        return 0.0, 1.0

    selected_set = set(selected_chunk_ids)
    intersection = selected_set.intersection(gold_relevant_ids)
    
    precision = len(intersection) / len(selected_chunk_ids)
    recall = len(intersection) / len(gold_relevant_ids)
    
    return precision, recall


def calculate_f1(precision: float, recall: float) -> float:
    """Calculates F1 score as the harmonic mean of precision and recall.

    Args:
        precision: Precision value in [0.0, 1.0].
        recall: Recall value in [0.0, 1.0].

    Returns:
        float: F1 score in [0.0, 1.0]. Returns 0.0 if both inputs are 0.
    """
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def calculate_ndcg(
    selected_chunk_ids: List[int],
    chunk_relevance_scores: dict,
    k: Optional[int] = None
) -> float:
    """Calculates Normalized Discounted Cumulative Gain using graded relevance scores.

    Measures how well the selected chunks capture highly relevant information,
    weighted by the position in which they were selected. Uses the ordering
    of selected_chunk_ids as the ranked list.

    Args:
        selected_chunk_ids: Ordered list of chunk IDs as selected by the router.
        chunk_relevance_scores: Dict mapping chunk_id -> relevance_score (0-3).
        k: Optional cutoff. If None, uses len(selected_chunk_ids).

    Returns:
        float: NDCG score in [0.0, 1.0]. Returns 0.0 if no relevant items exist.
    """
    if not selected_chunk_ids or not chunk_relevance_scores:
        return 0.0

    # Determine cutoff
    if k is None:
        k = len(selected_chunk_ids)
    k = min(k, len(selected_chunk_ids))

    # DCG of the selected ranking
    dcg = 0.0
    for i in range(k):
        cid = selected_chunk_ids[i]
        rel = chunk_relevance_scores.get(cid, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: sort all relevance scores descending, take top k
    all_scores = sorted(chunk_relevance_scores.values(), reverse=True)
    ideal_k = min(k, len(all_scores))
    idcg = 0.0
    for i in range(ideal_k):
        idcg += all_scores[i] / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_gold_retention(
    original_chunks: List[str],
    selected_chunks: List[str],
    gold_indices: List[int]
) -> float:
    """Legacy helper preserving backward compatibility for previous milestones.

    Checks what proportion of strings in original_chunks[gold_indices] are in selected_chunks.
    """
    if not gold_indices:
        return 1.0
        
    selected_set = set(selected_chunks)
    retained = 0
    for idx in gold_indices:
        if 0 <= idx < len(original_chunks):
            gold_sentence = original_chunks[idx]
            if gold_sentence in selected_set:
                retained += 1
                
    return retained / len(gold_indices)
