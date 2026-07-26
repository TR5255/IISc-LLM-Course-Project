import os
from typing import Any, Dict, List

class LegalDocumentLoader:
    """Placeholder loader for parsing legal agreement documents (e.g. NDAs, TOS)."""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_document(self, doc_id: str) -> Dict[str, Any]:
        """Loads a single document by ID. Returns simulated legal text."""
        # Simulated legal document text for baseline verification
        simulated_text = (
            f"--- CONFIDENTIALITY AGREEMENT ({doc_id}) ---\n"
            "1. Definition of Confidential Information. Confidential Information includes raw data and proprietary code.\n"
            "2. Obligations. Receiving party shall maintain secrecy and limit disclosure.\n"
            "3. Term. This agreement remains in effect for 3 years from signature.\n"
            "4. Governing Law. This agreement is governed by the laws of the State of California."
        )
        return {
            "doc_id": doc_id,
            "text": simulated_text,
            "metadata": {"type": "NDA", "jurisdiction": "California"}
        }

    def list_available_documents(self) -> List[str]:
        """Lists available document IDs."""
        return ["nda_sample_01", "tos_sample_02"]
