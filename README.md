# 🔬 Research Agent

An autonomous research agent powered by Claude that conducts comprehensive web research and produces structured Markdown reports with citations.

Point it at a topic, and it'll search the web, read pages, extract findings, validate quotes against sources, and write you a research document.

## ✨ Features

- 🎯 **Structured Research Briefs** - Define your topic, detail level, and specific questions in a simple JSON file
- 📚 **Wikipedia Bootstrapping** - Automatically gathers background context before diving into web research
- 🔍 **Smart Web Search** - Uses DuckDuckGo to find relevant sources (no API keys needed!)
- 🌐 **Browser Automation** - Playwright-powered browsing that handles JavaScript-heavy sites
- ✅ **Quote Validation** - Direct quotes are verified against source pages before being stored
- 🧠 **Vector Memory** - Stores findings with semantic embeddings for intelligent retrieval
- 💾 **Persistent Knowledge Stores** - Each session creates a queryable SQLite + vector database
- 🔌 **MCP Server** - Query your knowledge stores from Claude Desktop or any MCP-compatible client
- 🖥️ **Web Explorer** - Browse and explore your knowledge stores with a local web UI
- ⚡ **Parallel Execution** - Concurrent browser instances and async storage for speed
- 🛑 **Graceful Shutdown** - Ctrl+C saves your progress, double Ctrl+C force quits

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ConorCallanan2003/research-agent.git
cd research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install browser (required for web scraping)
playwright install chromium
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run your first research

```bash
python -m research_agent.main briefs/example_brief.json
```

## 📝 Creating Research Briefs

Research briefs are JSON files that tell the agent what to research. Here's an example:

```json
{
  "topic": "Large Language Model Agents",
  "detail_level": "moderate",
  "time_focus": {
    "current_weight": 0.7,
    "historical_weight": 0.3
  },
  "specific_questions": [
    "What are the main architectural patterns for LLM agents?",
    "How do tool-use capabilities work in modern LLM agents?",
    "What are the current limitations of LLM agents?"
  ],
  "excluded_sources": ["medium.com", "quora.com"],
  "max_sources": 15
}
```

### Brief Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | ✅ | Main research topic |
| `detail_level` | enum | | `overview`, `moderate`, or `comprehensive` (default: moderate) |
| `time_focus.current_weight` | float | | 0-1, emphasis on recent info (default: 0.5) |
| `time_focus.historical_weight` | float | | 0-1, emphasis on historical info (default: 0.5) |
| `specific_questions` | list[string] | | Specific questions to answer |
| `excluded_sources` | list[string] | | Domains to skip (e.g., "medium.com") |
| `max_sources` | int | | Max sources to consult (default: 20) |
| `finding_targets.*` | int | | Target counts for direct_quote, paraphrase, summary, synthesis |

## 🎮 CLI Options

```bash
python -m research_agent.main <brief.json> [options]

Options:
  -o, --output DIR      Output directory for the report (default: current dir)
  -v, --verbose         Show agent's thinking and reasoning
  -f, --fast            Use Haiku model (faster, cheaper, less thorough)
  --no-thinking         Disable extended thinking (faster but less capable)
```

## ⚙️ Configuration

Set environment variables in `.env`:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
BROWSER_HEADLESS=true           # Run browser in headless mode (default: false)
KNOWLEDGE_STORE_DIR=./stores    # Knowledge store directory
CLAUDE_MODEL=claude-sonnet-4-5-20250929
```

**Headless Mode:** Now fully supported with `playwright-stealth`! Yahoo search works reliably in headless mode with automated consent dialog handling.

## 🖥️ Knowledge Store Explorer

Browse your research findings with a local web UI:

```bash
python -m explorer.server
```

Then open http://localhost:8000 to:
- 📂 Browse all your knowledge stores
- 🔍 View findings with full metadata
- 🔗 Explore semantic neighbors of any finding
- 📚 See all sources used in a research session

## 🔌 MCP Server

Query your knowledge stores from any MCP-compatible client (Claude Desktop, other local LLM interfaces, etc.)

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "research-knowledge": {
      "command": "/path/to/research-agent/venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/research-agent"
    }
  }
}
```

**Available MCP Tools:**
- `list_knowledge_stores()` - List all your research stores
- `query_knowledge_store(store_name, query, k=10)` - Semantic search
- `get_store_statistics(store_name)` - Store metadata and stats

## 📦 Output

Each research session produces:

1. **📄 Markdown Report** (`{topic}_{timestamp}_{level}.md`)
   - Executive summary
   - Key findings
   - Detailed findings grouped by topic
   - Bibliography with access dates

2. **💾 Knowledge Store** (`knowledge_stores/{topic}_{timestamp}_{level}.db`)
   - SQLite database with all findings
   - HNSW vector index for semantic search
   - Full citation metadata

## 🏗️ Architecture

```
research_agent/
├── src/research_agent/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration
│   ├── models/              # Pydantic models (Brief, Finding, Citation)
│   ├── initial_research/    # Wikipedia bootstrapping
│   ├── planner/             # Research plan generation
│   ├── tools/
│   │   ├── browser.py       # Playwright web scraping
│   │   ├── memory.py        # Vector store interface
│   │   ├── finding_queue.py # Async storage queue
│   │   └── quote_validator.py # Direct quote verification
│   ├── embeddings/          # Qwen3 embedding model
│   ├── agent/               # Claude agent loop
│   │   ├── loop.py          # Main agentic loop
│   │   ├── system_prompt.py # Prompt engineering
│   │   └── tools_schema.py  # Tool definitions
│   └── output/              # Report rendering
├── explorer/                # Web UI for browsing stores
├── mcp_server/              # MCP server for Claude Desktop
├── knowledge_stores/        # Your research databases
└── briefs/                  # Research brief templates
```

## 💻 Requirements

- **Python 3.11+**
- **~8GB VRAM** (GPU) or **~16GB RAM** (CPU) for the embedding model (probably - tested on M1 Pro MacBook with 16GB RAM and it runs fine)
- **Anthropic API key** - Get one at [console.anthropic.com](https://console.anthropic.com)
- **Internet connection** - For web research

## 🙏 Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client |
| `playwright` | Browser automation |
| `torch` + `sentence-transformers` | Local embedding model |
| `hnswlib` | Vector similarity search |
| `pydantic` | Data validation |
| `wikipedia-api` | Initial research |
| `fastapi` + `uvicorn` | Explorer web server |
| `mcp` | Model Context Protocol server |
| `rich` | Beautiful terminal output |

## 🤝 Contributing

Contributions welcome! This project was built with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

## 📄 License

MIT
