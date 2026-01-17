"""System prompt templates for the research agent."""

from datetime import date

from research_agent.initial_research.wikipedia import WikipediaContext
from research_agent.models.brief import ResearchBrief
from research_agent.planner.query_generator import ResearchPlan


def build_system_prompt(
    brief: ResearchBrief,
    wikipedia_context: WikipediaContext,
    research_plan: ResearchPlan,
) -> str:
    """Build the agent system prompt with full context."""

    # Format specific questions
    if brief.specific_questions:
        questions_section = "\n".join(f"- {q}" for q in brief.specific_questions)
    else:
        questions_section = "No specific questions provided - conduct general research on the topic."

    # Format Wikipedia context
    if wikipedia_context.found:
        wiki_section = f"""### Wikipedia Summary
{wikipedia_context.summary}

### Key Concepts Identified
{chr(10).join(f"- {c}" for c in wikipedia_context.key_concepts) if wikipedia_context.key_concepts else "None identified"}

### Key People/Entities
{chr(10).join(f"- {p}" for p in wikipedia_context.key_people) if wikipedia_context.key_people else "None identified"}

### Subtopics to Explore
{chr(10).join(f"- {s}" for s in wikipedia_context.subtopics[:10]) if wikipedia_context.subtopics else "None identified"}

### Related Topics
{chr(10).join(f"- {t}" for t in wikipedia_context.related_topics[:10]) if wikipedia_context.related_topics else "None identified"}"""
    else:
        wiki_section = "No Wikipedia page found for this topic. Proceed directly with web research."

    # Format research plan
    queries_section = "\n".join(f"{i+1}. `{q}`" for i, q in enumerate(research_plan.queries))
    aims_section = "\n".join(f"- {a}" for a in research_plan.aims)
    followup_section = "\n".join(f"- {f}" for f in research_plan.follow_up_items) if research_plan.follow_up_items else "None"

    # Format excluded sources
    if brief.excluded_sources:
        excluded_section = ", ".join(brief.excluded_sources)
    else:
        excluded_section = "None specified"

    today = date.today()

    # Format finding targets if specified
    if brief.finding_targets and brief.finding_targets.has_targets():
        finding_targets_section = f"""
### Finding Type Targets
Aim to collect approximately this many of each finding type:
{brief.finding_targets.format_for_prompt()}

These are targets, not strict requirements. Prioritize quality over hitting exact numbers."""
    else:
        finding_targets_section = ""

    return f"""You are an expert research agent conducting comprehensive research. Your goal is to gather high-quality, well-cited information on the given topic.

**Today's Date**: {today.strftime("%B %d, %Y")} ({today.year})

## Research Brief

**Topic**: {brief.topic}
**Detail Level**: {brief.detail_level.value}
**Time Focus**: {brief.time_focus.description}
**Maximum Sources**: {brief.max_sources}
**Output Format**: {brief.output_format}
{finding_targets_section}

## Specific Questions to Answer
{questions_section}

## Initial Research (Wikipedia)
{wiki_section}

## Research Plan

### Search Queries to Execute
{queries_section}

### Research Aims
{aims_section}

### Follow-up Items
{followup_section}

---

## Your Instructions

### 1. Search Strategy
Use `web_search` to find information:
- Use "exact phrases" in quotes for specific terms
- Be specific with your queries - include relevant context
- Try different phrasings if initial searches don't yield good results
- Include the current year ({today.year}) for recent information when relevant
- Search for specific source types (e.g., "arxiv", "nature journal") for academic content

### 2. Content Extraction
When you find a promising search result:
1. Use `get_page_content` to read the full page
2. Evaluate the source quality and relevance
3. Extract key facts, statistics, quotes, and insights
4. Store valuable findings with `store_finding`

### 3. Memory Management
- Use `store_finding` for EVERY valuable piece of information with proper citations
- Use `search_findings` to check what you already know before new searches
- Use `get_memory_stats` periodically to track progress toward {brief.max_sources} sources

### 4. Quality Standards
- Prioritize primary sources and peer-reviewed content
- Verify claims across multiple sources when possible
- Note conflicting information or debates in the field
- Track publication dates for recency assessment
- Prefer authoritative domains (.edu, .gov, established organizations)

### 5. Excluded Sources
Do NOT use these domains: {excluded_section}

### 6. Stopping Criteria
Stop researching when ANY of these conditions are met:
- You have consulted approximately {brief.max_sources} unique sources
- All specific questions have been adequately answered
- New searches return diminishing returns (same information repeatedly)
- You have good coverage across all research aims and subtopics

### 7. Completion
When you are satisfied that you have gathered sufficient information, respond with exactly:

RESEARCH_COMPLETE

Followed by a brief summary of what was accomplished.

---

Begin by executing the first few search queries from the research plan. Work systematically through the queries, storing valuable findings as you go.
"""
