from typing import List, Dict, Any, Optional

class Chunk:
    """Represents a single segment (sentence/paragraph) of a document with research metadata."""
    def __init__(
        self,
        chunk_id: int,
        text: str,
        token_count: int,
        is_relevant: bool,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        relevance_score: Optional[int] = 0
    ):
        self.id = chunk_id
        self.text = text
        self.token_count = token_count
        self.is_relevant = is_relevant
        self.start_position = start_position
        self.end_position = end_position
        self.relevance_score = relevance_score  # Graded relevance: 0 to 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["id"],
            text=data["text"],
            token_count=data["token_count"],
            is_relevant=data["is_relevant"],
            start_position=data.get("start_position"),
            end_position=data.get("end_position"),
            relevance_score=data.get("relevance_score", 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "token_count": self.token_count,
            "is_relevant": self.is_relevant,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "relevance_score": self.relevance_score
        }


class BenchmarkQAItem:
    """Represents a single query-driven context evaluation item with provenance and difficulty."""
    def __init__(
        self,
        document_id: str,
        document_text: str,
        question: str,
        answer: str,
        chunks: List[Chunk],
        difficulty: str,
        provenance: Dict[str, str]
    ):
        self.document_id = document_id
        self.document_text = document_text
        self.question = question
        self.answer = answer
        self.chunks = chunks
        self.difficulty = difficulty  # "easy", "medium", or "hard"
        self.provenance = provenance  # source_type, source_name, license

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQAItem":
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        
        # Dynamically calculate/align start and end positions to prevent JSON drift
        doc_text = data["document_text"]
        search_ptr = 0
        for c in chunks:
            idx = doc_text.find(c.text, search_ptr)
            if idx != -1:
                c.start_position = idx
                c.end_position = idx + len(c.text)
                search_ptr = c.end_position
            else:
                # Fallback if not found in strict lookup
                c.start_position = c.start_position or 0
                c.end_position = c.end_position or len(c.text)

        return cls(
            document_id=data["document_id"],
            document_text=data["document_text"],
            question=data["question"],
            answer=data["answer"],
            chunks=chunks,
            difficulty=data["difficulty"],
            provenance=data["provenance"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_text": self.document_text,
            "question": self.question,
            "answer": self.answer,
            "chunks": [c.to_dict() for c in self.chunks],
            "difficulty": self.difficulty,
            "provenance": self.provenance
        }

