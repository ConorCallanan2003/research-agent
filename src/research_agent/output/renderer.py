"""Render research findings into final document format."""

from datetime import datetime
from pathlib import Path

from research_agent.models.brief import ResearchBrief
from research_agent.models.document import (
    BibliographyEntry,
    DocumentSection,
    ResearchDocument,
)
from research_agent.models.findings import Finding
from research_agent.tools.memory import MemoryTool


def render_document(
    brief: ResearchBrief,
    memory: MemoryTool,
    agent_summary: str,
    research_duration_seconds: float,
) -> ResearchDocument:
    """
    Render stored findings into a structured research document.

    Args:
        brief: The original research brief
        memory: Memory tool with stored findings
        agent_summary: Summary from the agent
        research_duration_seconds: How long the research took

    Returns:
        ResearchDocument with all findings organized
    """
    # Get all findings
    findings = memory.get_all_findings()
    stats = memory.get_statistics()

    # Build bibliography from unique sources
    bibliography = build_bibliography(findings)
    source_to_index = {entry.url: entry.index for entry in bibliography}

    # Organize findings into sections
    sections = organize_into_sections(findings, source_to_index)

    # Extract key findings (most important/relevant)
    key_findings = extract_key_findings(findings)

    # Generate executive summary
    executive_summary = generate_executive_summary(brief, findings, agent_summary)

    return ResearchDocument(
        title=f"Research Report: {brief.topic}",
        topic=brief.topic,
        generated_at=datetime.utcnow(),
        executive_summary=executive_summary,
        key_findings=key_findings,
        sections=sections,
        question_answers=[],  # Could be populated by agent analysis
        bibliography=bibliography,
        sources_consulted=stats["unique_sources"],
        detail_level=brief.detail_level.value,
        research_duration_seconds=research_duration_seconds,
    )


def build_bibliography(findings: list[Finding]) -> list[BibliographyEntry]:
    """Build deduplicated bibliography from findings."""
    seen_urls = set()
    entries = []
    index = 1

    for finding in findings:
        url = finding.citation.source_url
        if url not in seen_urls:
            seen_urls.add(url)
            entries.append(
                BibliographyEntry(
                    index=index,
                    title=finding.citation.title,
                    url=url,
                    author=finding.citation.author,
                    publication_date=finding.citation.publication_date,
                    accessed_date=finding.citation.accessed_date,
                )
            )
            index += 1

    return entries


def organize_into_sections(
    findings: list[Finding], source_to_index: dict[str, int]
) -> list[DocumentSection]:
    """Organize findings into logical sections based on relevance notes."""
    if not findings:
        return []

    # Group findings by their relevance notes (simplified approach)
    # A more sophisticated version would use clustering
    sections_dict: dict[str, list[tuple[Finding, int]]] = {}

    for finding in findings:
        # Use first few words of relevance notes as section key
        key = finding.relevance_notes.split(".")[0][:50] if finding.relevance_notes else "General"
        if key not in sections_dict:
            sections_dict[key] = []
        citation_idx = source_to_index.get(finding.citation.source_url, 0)
        sections_dict[key].append((finding, citation_idx))

    # Convert to sections
    sections = []
    for heading, items in sections_dict.items():
        content_parts = []
        citations = []

        for finding, citation_idx in items:
            content_parts.append(f"- {finding.text}")
            if citation_idx and citation_idx not in citations:
                citations.append(citation_idx)

        sections.append(
            DocumentSection(
                heading=heading,
                content="\n".join(content_parts),
                citations=citations,
            )
        )

    return sections


def extract_key_findings(findings: list[Finding], max_findings: int = 10) -> list[str]:
    """Extract the most important findings."""
    # Sort by confidence and take top N
    sorted_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)

    key_findings = []
    for finding in sorted_findings[:max_findings]:
        # Truncate if too long
        text = finding.text
        if len(text) > 200:
            text = text[:200] + "..."
        key_findings.append(text)

    return key_findings


def generate_executive_summary(
    brief: ResearchBrief, findings: list[Finding], agent_summary: str
) -> str:
    """Generate an executive summary."""
    num_findings = len(findings)
    num_sources = len(set(f.citation.source_url for f in findings))

    summary_parts = [
        f"This research report covers the topic of **{brief.topic}** at a {brief.detail_level.value} level of detail.",
        f"",
        f"A total of **{num_findings} findings** were gathered from **{num_sources} unique sources**.",
        f"",
    ]

    if agent_summary and "RESEARCH_COMPLETE" in agent_summary:
        # Extract the summary part after RESEARCH_COMPLETE
        parts = agent_summary.split("RESEARCH_COMPLETE")
        if len(parts) > 1 and parts[1].strip():
            summary_parts.append("### Agent Summary")
            summary_parts.append(parts[1].strip())

    return "\n".join(summary_parts)


def render_to_markdown(document: ResearchDocument) -> str:
    """Render a ResearchDocument to Markdown format."""
    lines = [
        f"# {document.title}",
        "",
        f"*Generated: {document.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Detail Level: {document.detail_level} | "
        f"Sources: {document.sources_consulted} | "
        f"Duration: {document.research_duration_seconds:.1f}s*",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        document.executive_summary,
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    for i, finding in enumerate(document.key_findings, 1):
        lines.append(f"{i}. {finding}")

    lines.extend(["", "---", "", "## Detailed Findings", ""])

    for section in document.sections:
        lines.append(f"### {section.heading}")
        lines.append("")
        lines.append(section.content)
        if section.citations:
            citation_refs = ", ".join(f"[{c}]" for c in section.citations)
            lines.append(f"\n*Sources: {citation_refs}*")
        lines.append("")

    if document.question_answers:
        lines.extend(["---", "", "## Questions Answered", ""])
        for qa in document.question_answers:
            lines.append(f"### Q: {qa.question}")
            lines.append("")
            lines.append(f"**Answer** ({qa.confidence}): {qa.answer}")
            if qa.supporting_citations:
                refs = ", ".join(f"[{c}]" for c in qa.supporting_citations)
                lines.append(f"\n*Supporting sources: {refs}*")
            lines.append("")

    lines.extend(["---", "", "## Bibliography", ""])

    for entry in document.bibliography:
        parts = []
        if entry.author:
            parts.append(entry.author)
        if entry.publication_date:
            parts.append(f"({entry.publication_date})")
        parts.append(f'"{entry.title}"')
        parts.append(f"Retrieved from {entry.url}")
        parts.append(f"Accessed {entry.accessed_date[:10]}")

        lines.append(f"{entry.index}. " + ". ".join(parts) + ".")

    return "\n".join(lines)


def save_document(document: ResearchDocument, output_path: str | Path) -> None:
    """Save document to a markdown file."""
    path = Path(output_path)
    markdown = render_to_markdown(document)
    path.write_text(markdown, encoding="utf-8")
