"""Models for research findings and citations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# Types of findings the agent can store
FindingType = Literal["direct_quote", "paraphrase", "summary", "synthesis"]


@dataclass
class Citation:
    """Citation metadata for a research finding."""

    source_url: str
    title: str
    author: str | None = None
    publication_date: str | None = None
    accessed_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage in hnsqlite metadata."""
        return {
            "source_url": self.source_url,
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "accessed_date": self.accessed_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Citation":
        """Create Citation from dictionary."""
        return cls(
            source_url=data.get("source_url", ""),
            title=data.get("title", ""),
            author=data.get("author"),
            publication_date=data.get("publication_date"),
            accessed_date=data.get("accessed_date", datetime.utcnow().isoformat()),
        )

    def format_bibliography(self, index: int) -> str:
        """Format as a bibliography entry."""
        parts = []
        if self.author:
            parts.append(self.author)
        if self.publication_date:
            parts.append(f"({self.publication_date})")
        parts.append(f'"{self.title}"')
        parts.append(f"Retrieved from {self.source_url}")
        parts.append(f"Accessed {self.accessed_date[:10]}")
        return f"{index}. " + ". ".join(parts) + "."


@dataclass
class Finding:
    """A research finding with citation."""

    text: str
    citation: Citation
    relevance_notes: str
    finding_type: FindingType = "paraphrase"  # direct_quote, paraphrase, summary, synthesis
    confidence: float = 1.0  # 0-1 confidence score
    doc_id: str | None = None

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for hnsqlite storage."""
        metadata = self.citation.to_dict()
        metadata["relevance_notes"] = self.relevance_notes
        metadata["finding_type"] = self.finding_type
        metadata["confidence"] = self.confidence
        return metadata

    @classmethod
    def from_storage(
        cls, text: str, metadata: dict[str, Any], doc_id: str | None = None
    ) -> "Finding":
        """Create Finding from hnsqlite storage format."""
        citation = Citation.from_dict(metadata)
        return cls(
            text=text,
            citation=citation,
            relevance_notes=metadata.get("relevance_notes", ""),
            finding_type=metadata.get("finding_type", "paraphrase"),
            confidence=metadata.get("confidence", 1.0),
            doc_id=doc_id,
        )
