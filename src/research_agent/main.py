"""CLI entry point for the research agent."""

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from research_agent.agent.loop import ResearchAgent
from research_agent.config import Config
from research_agent.embeddings.qwen_embedder import QwenEmbedder
from research_agent.initial_research.wikipedia import fetch_wikipedia_context
from research_agent.models.brief import ResearchBrief
from research_agent.output.renderer import render_document, render_to_markdown, save_document
from research_agent.planner.query_generator import format_plan_for_display, generate_research_plan
from research_agent.tools.browser import BrowserTool
from research_agent.tools.memory import MemoryTool, generate_store_name

console = Console()


class ShutdownHandler:
    """Handles graceful shutdown on Ctrl+C."""

    def __init__(self):
        self._interrupt_count = 0
        self._agent: ResearchAgent | None = None

    def set_agent(self, agent: ResearchAgent) -> None:
        """Set the agent to notify on shutdown."""
        self._agent = agent

    def handle_signal(self, signum, frame) -> None:
        """Handle interrupt signal."""
        self._interrupt_count += 1

        if self._interrupt_count == 1:
            console.print("\n[yellow]Interrupt received. Press Ctrl+C again to force quit.[/yellow]")
            if self._agent:
                self._agent.request_shutdown()
        else:
            console.print("\n[red]Force quitting...[/red]")
            sys.exit(1)


async def run_research(brief_path: str, output_dir: str | None = None, verbose: bool = False, fast: bool = False, thinking: bool = True) -> None:
    """Run the research agent with a given brief."""

    # Set up graceful shutdown handler
    shutdown_handler = ShutdownHandler()
    signal.signal(signal.SIGINT, shutdown_handler.handle_signal)

    # Validate config
    try:
        Config.ensure_valid()
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red]\n{e}")
        sys.exit(1)

    # Load and validate brief
    console.print(Panel(f"Loading research brief from: {brief_path}"))
    try:
        brief = ResearchBrief.from_json_file(brief_path)
    except Exception as e:
        console.print(f"[red]Error loading brief:[/red] {e}")
        sys.exit(1)

    console.print(f"[green]Topic:[/green] {brief.topic}")
    console.print(f"[green]Detail Level:[/green] {brief.detail_level.value}")
    console.print(f"[green]Max Sources:[/green] {brief.max_sources}")

    if brief.specific_questions:
        console.print("[green]Questions:[/green]")
        for q in brief.specific_questions:
            console.print(f"  - {q}")

    # Preload embedding model (downloads weights if needed)
    console.print()
    console.print(Panel("Loading Embedding Model"))

    embedder = QwenEmbedder()
    if not embedder.is_loaded:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Loading {Config.EMBEDDING_MODEL} (this may download ~8GB on first run)...",
                total=None,
            )
            embedder._load_model()
            progress.remove_task(task)

    console.print(f"[green]Model loaded on device:[/green] {embedder.device}")

    # Step 1: Initial research (Wikipedia)
    console.print()
    console.print(Panel("Step 1: Initial Research (Wikipedia)"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching Wikipedia context...", total=None)
        wikipedia_context = fetch_wikipedia_context(brief.topic)
        progress.remove_task(task)

    if wikipedia_context.found:
        console.print(f"[green]Found Wikipedia page:[/green] {wikipedia_context.title}")
        console.print(f"[dim]Summary: {wikipedia_context.summary[:200]}...[/dim]")
    else:
        console.print("[yellow]No Wikipedia page found. Proceeding with web search.[/yellow]")

    # Step 2: Generate research plan
    console.print()
    console.print(Panel("Step 2: Generating Research Plan"))

    research_plan = generate_research_plan(brief, wikipedia_context)
    console.print(format_plan_for_display(research_plan))

    # Step 3: Initialize tools
    console.print()
    console.print(Panel("Step 3: Initializing Tools"))

    store_name = generate_store_name(brief.topic, brief.detail_level.value)
    console.print(f"[green]Knowledge store:[/green] {store_name}")

    memory_tool = MemoryTool(store_name=store_name)
    memory_tool.initialize()

    browser_tool = BrowserTool()

    # Step 4: Run research
    console.print()
    console.print(Panel("Step 4: Running Research Agent"))

    start_time = time.time()

    # Select model
    model = Config.CLAUDE_FAST_MODEL if fast else Config.CLAUDE_MODEL
    console.print(f"[green]Model:[/green] {model}")
    console.print(f"[green]Thinking:[/green] {'enabled' if thinking else 'disabled'}")

    async with browser_tool:
        agent = ResearchAgent(browser_tool=browser_tool, memory_tool=memory_tool, model=model, verbose=verbose, thinking=thinking)
        shutdown_handler.set_agent(agent)
        agent_summary = await agent.run(brief, wikipedia_context, research_plan)

    elapsed_time = time.time() - start_time

    console.print(f"\n[green]Research completed in {elapsed_time:.1f} seconds[/green]")
    console.print(f"[green]Turns used: {agent.turn_count}[/green]")

    # Step 5: Generate output document
    console.print()
    console.print(Panel("Step 5: Generating Research Document"))

    document = render_document(
        brief=brief,
        memory=memory_tool,
        agent_summary=agent_summary,
        research_duration_seconds=elapsed_time,
    )

    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / f"{store_name}.md"
    else:
        output_path = Path(f"{store_name}.md")

    save_document(document, output_path)
    console.print(f"[green]Document saved to:[/green] {output_path}")

    # Print summary
    console.print()
    console.print(Panel("Research Summary"))
    console.print(f"  Findings stored: {document.sources_consulted}")
    console.print(f"  Key findings: {len(document.key_findings)}")
    console.print(f"  Bibliography entries: {len(document.bibliography)}")
    console.print(f"  Knowledge store: {memory_tool.store_path}.db")

    # Token usage
    console.print()
    console.print(Panel("Token Usage"))
    console.print(f"  Input tokens:  {agent.total_input_tokens:,}")
    console.print(f"  Output tokens: {agent.total_output_tokens:,}")
    console.print(f"  Total tokens:  {agent.total_tokens:,}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Research Agent - Autonomous web research powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  research-agent briefs/my_topic.json
  research-agent briefs/my_topic.json --output ./reports/

Example brief JSON:
{
  "topic": "Quantum Computing Applications in Drug Discovery",
  "detail_level": "comprehensive",
  "time_focus": {"current_weight": 0.7, "historical_weight": 0.3},
  "specific_questions": [
    "What quantum algorithms are most promising for molecular simulation?",
    "Which pharmaceutical companies are investing in quantum computing?"
  ],
  "max_sources": 20
}
        """,
    )

    parser.add_argument(
        "brief",
        type=str,
        help="Path to the research brief JSON file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for the research document (default: current directory)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show agent's reasoning between tool calls",
    )

    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Use faster/cheaper Haiku model instead of Sonnet",
    )

    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended thinking (enabled by default)",
    )

    args = parser.parse_args()

    # Run the async main function
    asyncio.run(run_research(args.brief, args.output, args.verbose, args.fast, not args.no_thinking))


if __name__ == "__main__":
    main()
