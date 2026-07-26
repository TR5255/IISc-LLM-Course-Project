import re
from typing import List

class DocumentSplitter:
    """Utility to split long documents into manageable chunks (e.g. sentences or paragraphs)."""
    def __init__(self, method: str = "sentence"):
        self.method = method

    def split(self, text: str) -> List[str]:
        """Splits document text into chunks based on the configured method."""
        if self.method == "paragraph":
            # Split by double newline or carriage return
            chunks = re.split(r'\n\s*\n', text)
        else:
            # Simple sentence boundaries splitter
            chunks = re.split(r'(?<=[.!?])\s+', text)
            
        return [c.strip() for c in chunks if c.strip()]
