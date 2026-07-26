from typing import List

def calculate_compression_ratio(original_text: str, compressed_text: str) -> float:
    """Calculates the character-level compression ratio.

    Returns:
        float: Remaining character ratio (compressed / original).
        A lower value indicates more context was removed.
    """
    orig_len = len(original_text)
    if orig_len == 0:
        return 0.0
    return len(compressed_text) / orig_len

def calculate_token_compression_ratio(original_chunks: List[str], compressed_chunks: List[str]) -> float:
    """Calculates approximation of token-level compression ratio using simple whitespace splitting."""
    orig_tokens = sum(len(c.split()) for c in original_chunks)
    if orig_tokens == 0:
        return 0.0
    comp_tokens = sum(len(c.split()) for c in compressed_chunks)
    
    return comp_tokens / orig_tokens
