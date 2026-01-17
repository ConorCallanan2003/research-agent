"""Research brief schema and validation."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DetailLevel(str, Enum):
    """Level of detail for research."""

    OVERVIEW = "overview"  # High-level summary
    MODERATE = "moderate"  # Balanced depth
    COMPREHENSIVE = "comprehensive"  # Deep dive


class FindingTargets(BaseModel):
    """Optional targets for finding types."""

    direct_quote: int | None = Field(
        default=None,
        ge=0,
        description="Target number of direct quotes",
    )
    paraphrase: int | None = Field(
        default=None,
        ge=0,
        description="Target number of paraphrased findings",
    )
    summary: int | None = Field(
        default=None,
        ge=0,
        description="Target number of summaries",
    )
    synthesis: int | None = Field(
        default=None,
        ge=0,
        description="Target number of synthesis findings",
    )

    def has_targets(self) -> bool:
        """Check if any targets are set."""
        return any([
            self.direct_quote is not None,
            self.paraphrase is not None,
            self.summary is not None,
            self.synthesis is not None,
        ])

    def format_for_prompt(self) -> str:
        """Format targets for inclusion in agent prompt."""
        targets = []
        if self.direct_quote is not None:
            targets.append(f"- Direct quotes: {self.direct_quote}")
        if self.paraphrase is not None:
            targets.append(f"- Paraphrases: {self.paraphrase}")
        if self.summary is not None:
            targets.append(f"- Summaries: {self.summary}")
        if self.synthesis is not None:
            targets.append(f"- Syntheses: {self.synthesis}")
        return "\n".join(targets)


class TimeFocus(BaseModel):
    """Balance between current and historical information."""

    current_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for current/recent information (0-1)",
    )
    historical_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for historical information (0-1)",
    )

    @model_validator(mode="after")
    def normalize_weights(self) -> "TimeFocus":
        """Ensure weights sum to 1.0."""
        total = self.current_weight + self.historical_weight
        if total > 0:
            self.current_weight = self.current_weight / total
            self.historical_weight = self.historical_weight / total
        return self

    @property
    def description(self) -> str:
        """Human-readable description of the time focus."""
        if self.current_weight > 0.7:
            return "Focus on recent developments and current state"
        elif self.historical_weight > 0.7:
            return "Focus on historical context and evolution"
        return "Balance current and historical perspectives"


class ResearchBrief(BaseModel):
    """Schema for research brief JSON input."""

    topic: str = Field(
        min_length=3,
        description="Main research topic",
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.MODERATE,
        description="Depth of research required",
    )
    time_focus: TimeFocus = Field(
        default_factory=TimeFocus,
        description="Balance between current and historical focus",
    )
    specific_questions: list[str] = Field(
        default_factory=list,
        description="Specific questions to answer during research",
    )
    excluded_sources: list[str] = Field(
        default_factory=list,
        description="Domains or sources to exclude from research",
    )
    output_format: Literal["markdown", "json"] = Field(
        default="markdown",
        description="Desired output format for the research document",
    )
    max_sources: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum number of sources to consult",
    )
    finding_targets: FindingTargets | None = Field(
        default=None,
        description="Optional targets for number of findings by type",
    )

    @classmethod
    def from_json_file(cls, path: str) -> "ResearchBrief":
        """Load and validate a research brief from a JSON file."""
        import json
        from pathlib import Path

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Research brief not found: {path}")

        with open(file_path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    def get_search_date_preference(self) -> str | None:
        """Get Google date filter based on time focus."""
        if self.time_focus.current_weight > 0.7:
            return "y"  # Past year
        elif self.time_focus.current_weight > 0.5:
            return "y2"  # Past 2 years (custom)
        return None  # No date restriction
