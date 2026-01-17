"""Tool definitions for Claude API."""

# Tool schemas for Claude's tool_use feature
TOOLS = [
    {
        "name": "web_search",
        "description": """Search the web for information using DuckDuckGo. Returns a list of search results with titles, URLs, and snippets.

Tips for effective searches:
- Use "exact phrase" in quotes for specific terms
- Be specific with your queries
- Try different phrasings if initial search doesn't yield good results
- Combine topic with specific aspects you're researching

Examples:
- "Cerebras AI" chip architecture
- "quantum computing" drug discovery applications
- "machine learning" healthcare 2024""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "num_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results to return (default: 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_page_content",
        "description": """Navigate to a URL and extract its text content. Handles JavaScript-rendered pages.

Use this after finding promising URLs from web_search. The content is automatically cleaned to remove navigation, ads, and other non-content elements.

Returns the page title, main text content (truncated if very long), and a list of links found on the page.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to visit",
                },
                "wait_for_js": {
                    "type": "boolean",
                    "default": True,
                    "description": "Wait for JavaScript to render (default: true)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "store_finding",
        "description": """Store a research finding in memory with citation metadata for later retrieval.

Use this whenever you discover:
- Important facts, statistics, or data points
- Key quotes from experts or sources
- Explanations of concepts
- Definitions
- Historical information
- Current developments

Always provide complete citation information and explain why this finding is relevant to the research.

IMPORTANT: Classify each finding by type:
- "direct_quote": Exact words from the source. MUST be word-for-word text that appears in the source - this will be validated against the source URL and rejected if not found. Use only when you need the precise wording (statistics, definitions, expert statements). Copy the text exactly as it appears.
- "paraphrase": Restating specific information in different words while preserving the meaning. Be thorough - include all relevant details, context, and nuance from the original. Aim for 3-5 sentences minimum.
- "summary": Condensing a longer passage or section into key points. Be comprehensive - capture all the important information, not just the headline. Include specific details, examples, and data points. Aim for a substantial paragraph (5-8 sentences).
- "synthesis": Combining information from the current source with prior knowledge or other sources. Explain the connections and implications in detail.

NOTE: Direct quotes are automatically validated against the source URL. If the exact text cannot be found, the tool will return an error. Use paraphrase or summary if you're not copying text verbatim.

IMPORTANT: Findings should be detailed and substantive. Avoid superficial one-sentence summaries. Each finding should be self-contained and provide enough context to be useful on its own.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The finding text to store. Should be a self-contained piece of information. For direct_quote type: copy the EXACT text from the source - it will be validated and rejected if not found verbatim.",
                },
                "finding_type": {
                    "type": "string",
                    "enum": ["direct_quote", "paraphrase", "summary", "synthesis"],
                    "description": "Type of finding: direct_quote (exact words), paraphrase (restated), summary (condensed), or synthesis (combined with other knowledge)",
                },
                "source_url": {
                    "type": "string",
                    "description": "URL where the finding was found",
                },
                "title": {
                    "type": "string",
                    "description": "Page or article title",
                },
                "author": {
                    "type": "string",
                    "description": "Author if known (optional)",
                },
                "publication_date": {
                    "type": "string",
                    "description": "Publication date if known, in any reasonable format (optional)",
                },
                "relevance_notes": {
                    "type": "string",
                    "description": "Brief note on why this finding is relevant to the research",
                },
            },
            "required": ["text", "finding_type", "source_url", "title", "relevance_notes"],
        },
    },
    {
        "name": "search_findings",
        "description": """Semantic search through stored findings. Use this to:

1. Check what you already know before searching the web (avoid redundant searches)
2. Find related information you've already gathered
3. Verify if you have coverage of a topic
4. Connect information across different sources

The search is semantic, so use natural language queries.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results to return (default: 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_memory_stats",
        "description": """Get statistics about stored findings.

Returns:
- Total number of findings stored
- Number of unique sources consulted
- List of source URLs

Use this to track your research progress and ensure you're meeting the research brief requirements.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
