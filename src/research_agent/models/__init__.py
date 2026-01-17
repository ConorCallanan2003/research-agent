"""Data models for the research agent."""

from .brief import ResearchBrief, TimeFocus, DetailLevel
from .findings import Finding, Citation
from .document import ResearchDocument, DocumentSection, QuestionAnswer, BibliographyEntry

__all__ = [
    "ResearchBrief",
    "TimeFocus",
    "DetailLevel",
    "Finding",
    "Citation",
    "ResearchDocument",
    "DocumentSection",
    "QuestionAnswer",
    "BibliographyEntry",
]
