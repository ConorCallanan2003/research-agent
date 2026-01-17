"""Models for the final research document."""

from datetime import datetime

from pydantic import BaseModel, Field


class BibliographyEntry(BaseModel):
    """A bibliography entry."""

    index: int
    title: str
    url: str
    author: str | None = None
    publication_date: str | None = None
    accessed_date: str


class QuestionAnswer(BaseModel):
    """An answered research question."""

    question: str
    answer: str
    confidence: str = Field(description="high, medium, or low")
    supporting_citations: list[int] = Field(
        default_factory=list, description="Indices into bibliography"
    )


class DocumentSection(BaseModel):
    """A section of the research document."""

    heading: str
    content: str  # Markdown content
    citations: list[int] = Field(
        default_factory=list, description="Indices into bibliography"
    )


class ResearchDocument(BaseModel):
    """Final research document structure."""

    title: str
    topic: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Summary sections
    executive_summary: str
    key_findings: list[str]

    # Main content
    sections: list[DocumentSection]

    # Specific questions answered
    question_answers: list[QuestionAnswer] = Field(default_factory=list)

    # Sources and citations
    bibliography: list[BibliographyEntry]

    # Metadata
    sources_consulted: int
    detail_level: str
    research_duration_seconds: float
