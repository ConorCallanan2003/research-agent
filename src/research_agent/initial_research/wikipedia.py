"""Wikipedia-based initial research for gathering background context."""

import re
from dataclasses import dataclass

import wikipediaapi


@dataclass
class WikipediaContext:
    """Context gathered from Wikipedia."""

    title: str
    summary: str
    key_concepts: list[str]
    key_people: list[str]
    subtopics: list[str]
    related_topics: list[str]
    url: str
    found: bool = True


def fetch_wikipedia_context(topic: str) -> WikipediaContext:
    """
    Fetch Wikipedia page for a topic to get high-level context.

    Args:
        topic: The research topic to look up

    Returns:
        WikipediaContext with extracted information
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="ResearchAgent/1.0 (research-agent@example.com)",
        language="en",
    )

    # Try to find the page
    page = wiki.page(topic)

    if not page.exists():
        # Try with title case
        page = wiki.page(topic.title())

    if not page.exists():
        # Try search-like approach with underscores
        page = wiki.page(topic.replace(" ", "_"))

    if not page.exists():
        return WikipediaContext(
            title=topic,
            summary=f"No Wikipedia page found for '{topic}'. Research will proceed with web search.",
            key_concepts=[],
            key_people=[],
            subtopics=[],
            related_topics=[],
            url="",
            found=False,
        )

    # Extract summary (first few paragraphs)
    summary = page.summary
    if len(summary) > 2000:
        summary = summary[:2000] + "..."

    # Extract section titles as subtopics
    subtopics = [section.title for section in page.sections[:10]]

    # Extract key concepts from the summary using simple heuristics
    key_concepts = extract_key_concepts(summary)

    # Extract potential people/entities (capitalized multi-word phrases)
    key_people = extract_key_people(summary)

    # Get related/linked topics
    related_topics = list(page.links.keys())[:20]

    return WikipediaContext(
        title=page.title,
        summary=summary,
        key_concepts=key_concepts,
        key_people=key_people,
        subtopics=subtopics,
        related_topics=related_topics,
        url=page.fullurl,
        found=True,
    )


def extract_key_concepts(text: str) -> list[str]:
    """Extract key concepts from text using simple heuristics."""
    concepts = []

    # Look for phrases in quotes
    quoted = re.findall(r'"([^"]+)"', text)
    concepts.extend(quoted[:5])

    # Look for phrases with "known as", "called", "referred to as"
    known_as = re.findall(
        r"(?:known as|called|referred to as|termed)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Za-z]+)*)",
        text,
    )
    concepts.extend(known_as[:5])

    # Look for capitalized phrases (potential proper nouns/concepts)
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    # Filter out common false positives
    stop_phrases = {"The", "This", "These", "That", "Those", "In", "On", "At", "For"}
    caps = [c for c in caps if c.split()[0] not in stop_phrases]
    concepts.extend(caps[:10])

    # Deduplicate while preserving order
    seen = set()
    unique_concepts = []
    for c in concepts:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique_concepts.append(c)

    return unique_concepts[:15]


def extract_key_people(text: str) -> list[str]:
    """Extract potential person names from text."""
    # Pattern for names (First Last, or First Middle Last)
    name_pattern = r"\b([A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+)\b"
    names = re.findall(name_pattern, text)

    # Filter out common false positives
    stop_names = {
        "United States",
        "New York",
        "Los Angeles",
        "San Francisco",
        "World War",
        "North America",
        "South America",
        "European Union",
    }
    names = [n for n in names if n not in stop_names]

    # Deduplicate
    seen = set()
    unique_names = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique_names.append(n)

    return unique_names[:10]
