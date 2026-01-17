# Roadmap & Future Improvements

This document tracks ideas for future improvements to the research agent.

## Planned Improvements

### Haiku-Powered Browser Agent

**Priority:** Medium
**Complexity:** Medium

Replace the simple `get_page_content` function with a Haiku-orchestrated browser agent that can:

- Handle cookie consent dialogs and popups intelligently
- Navigate complex pages (click "read more", expand sections, etc.)
- Work around soft paywalls
- Decide what content is actually useful on a page
- Recover gracefully from errors

**Architecture:**
```
web_search (DuckDuckGo) → get URLs
    ↓
get_page_content (Haiku agent) → intelligently extracts content
    ↓
store_finding → saves to memory
```

The main research agent (Sonnet/Opus) would still orchestrate overall research, but Haiku handles lower-level browser interactions for each page visit. This keeps costs down while adding flexibility.

---

## Ideas Backlog

*Add new ideas below as they come up*

### Search Improvements
- [ ] Add fallback search engines (Bing, Brave Search) if DuckDuckGo fails
- [ ] Implement search result caching to avoid re-fetching
- [ ] Add support for academic search (Semantic Scholar API, arXiv API)

### Memory & Knowledge Store
- [ ] Add ability to merge multiple knowledge stores
- [ ] Implement knowledge store versioning
- [ ] Add export to different formats (PDF, DOCX, Notion)

### Research Quality
- [ ] Add source credibility scoring
- [ ] Implement fact cross-referencing across sources
- [ ] Add citation network analysis

### Performance
- [ ] Parallel page fetching
- [ ] Streaming embeddings generation
- [ ] Incremental research (resume from where you left off)

### User Experience
- [ ] Web UI for creating briefs and viewing results
- [ ] Real-time progress dashboard
- [ ] Interactive research mode (human-in-the-loop)

---

## Completed

- [x] Switch from Google to DuckDuckGo (avoid CAPTCHAs)
- [x] Custom SQLite + hnswlib storage (replaced hnsqlite dependency)
- [x] Knowledge Store Explorer UI (FastAPI + htmx)
