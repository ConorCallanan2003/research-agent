"""Research plan generation with search queries."""

from dataclasses import dataclass

from research_agent.initial_research.wikipedia import WikipediaContext
from research_agent.models.brief import DetailLevel, ResearchBrief


@dataclass
class ResearchPlan:
    """A structured research plan."""

    queries: list[str]  # Search queries
    aims: list[str]  # Research objectives
    follow_up_items: list[str]  # Items from initial research needing investigation


def generate_research_plan(
    brief: ResearchBrief, wikipedia_context: WikipediaContext
) -> ResearchPlan:
    """
    Generate a research plan based on brief and initial Wikipedia research.

    Args:
        brief: The research brief
        wikipedia_context: Context from Wikipedia

    Returns:
        ResearchPlan with queries, aims, and follow-up items
    """
    queries = []
    aims = []
    follow_up_items = []

    topic = brief.topic

    # Generate base queries based on detail level
    if brief.detail_level == DetailLevel.OVERVIEW:
        queries.extend(
            [
                f'"{topic}" overview introduction',
                f'"{topic}" explained',
                f'"{topic}" basics fundamentals',
                f'what is "{topic}"',
            ]
        )
        aims.append(f"Understand the basic concepts and fundamentals of {topic}")

    elif brief.detail_level == DetailLevel.MODERATE:
        queries.extend(
            [
                f'"{topic}" comprehensive guide',
                f'"{topic}" in-depth analysis',
                f'"{topic}" research',
                f'"{topic}" how it works',
            ]
        )
        aims.append(f"Develop a thorough understanding of {topic}")
        aims.append("Identify key debates and perspectives")

    else:  # COMPREHENSIVE
        queries.extend(
            [
                f'"{topic}" academic research',
                f'"{topic}" scholarly analysis',
                f'"{topic}" technical deep dive',
                f'"{topic}" systematic review',
                f'"{topic}" scientific paper',
            ]
        )
        aims.append(f"Achieve expert-level understanding of {topic}")
        aims.append("Analyze primary sources and academic literature")
        aims.append("Identify gaps in current knowledge")

    # Add time-focused queries based on time_focus
    if brief.time_focus.current_weight > 0.6:
        queries.extend(
            [
                f'"{topic}" 2024 2025 latest',
                f'"{topic}" recent developments news',
                f'"{topic}" current state',
            ]
        )
        aims.append("Focus on recent developments and current state")
    elif brief.time_focus.historical_weight > 0.6:
        queries.extend(
            [
                f'"{topic}" history origins',
                f'"{topic}" evolution timeline',
                f'"{topic}" historical development',
            ]
        )
        aims.append("Understand historical context and evolution")
    else:
        queries.append(f'"{topic}" history and current state')
        aims.append("Balance historical context with current developments")

    # Add queries for specific questions
    for question in brief.specific_questions:
        # Convert question to search query
        query = question.rstrip("?").replace("What is", "").replace("How does", "")
        query = query.replace("Why", "").replace("When", "").strip()
        queries.append(f'"{topic}" {query}')
        aims.append(f"Answer: {question}")

    # Generate follow-up items from Wikipedia context
    if wikipedia_context.found:
        # Add queries for key concepts
        for concept in wikipedia_context.key_concepts[:5]:
            queries.append(f'"{topic}" "{concept}"')
            follow_up_items.append(f"Investigate: {concept}")

        # Add queries for key people
        for person in wikipedia_context.key_people[:3]:
            queries.append(f'"{person}" "{topic}"')
            follow_up_items.append(f"Research contributions of: {person}")

        # Add queries for subtopics
        for subtopic in wikipedia_context.subtopics[:5]:
            if subtopic.lower() not in ["see also", "references", "external links", "notes", "further reading"]:
                queries.append(f'"{topic}" "{subtopic}"')
                follow_up_items.append(f"Explore subtopic: {subtopic}")

    # Add source-type queries for comprehensive research
    if brief.detail_level == DetailLevel.COMPREHENSIVE:
        queries.extend(
            [
                f'"{topic}" arxiv',
                f'"{topic}" nature journal',
                f'"{topic}" IEEE',
            ]
        )

    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    return ResearchPlan(
        queries=unique_queries,
        aims=aims,
        follow_up_items=follow_up_items,
    )


def format_plan_for_display(plan: ResearchPlan) -> str:
    """Format a research plan for human-readable display."""
    lines = ["## Research Plan", ""]

    lines.append("### Search Queries")
    for i, query in enumerate(plan.queries, 1):
        lines.append(f"{i}. `{query}`")
    lines.append("")

    lines.append("### Research Aims")
    for aim in plan.aims:
        lines.append(f"- {aim}")
    lines.append("")

    if plan.follow_up_items:
        lines.append("### Follow-up Items")
        for item in plan.follow_up_items:
            lines.append(f"- {item}")

    return "\n".join(lines)
