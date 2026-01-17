"""Main research agent loop with Claude tool use."""

import asyncio
import json
from typing import Any

import anthropic
from rich.console import Console
from rich.panel import Panel

from research_agent.agent.system_prompt import build_system_prompt
from research_agent.agent.tools_schema import TOOLS
from research_agent.config import Config
from research_agent.initial_research.wikipedia import WikipediaContext
from research_agent.models.brief import ResearchBrief
from research_agent.planner.query_generator import ResearchPlan
from research_agent.tools.browser import BrowserTool
from research_agent.tools.finding_queue import FindingQueue, StorageTask
from research_agent.tools.memory import MemoryTool
from research_agent.tools.quote_validator import validate_direct_quote

console = Console()


class ResearchAgent:
    """Main research agent with tool-use loop."""

    def __init__(
        self,
        browser_tool: BrowserTool,
        memory_tool: MemoryTool,
        model: str | None = None,
        verbose: bool = False,
        thinking: bool = True,
    ):
        self._client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self._browser = browser_tool
        self._memory = memory_tool
        self._finding_queue = FindingQueue(memory_tool)
        self._model = model or Config.CLAUDE_MODEL
        self._messages: list[dict] = []
        self._turn_count = 0
        self._verbose = verbose
        self._thinking = thinking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._shutdown_requested = False

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the agent."""
        self._shutdown_requested = True
        console.print("\n[yellow]Shutdown requested. Finishing current operation...[/yellow]")

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool and return the result."""
        try:
            if tool_name == "web_search":
                results = await self._browser.web_search(
                    query=tool_input["query"],
                    num_results=tool_input.get("num_results", 10),
                )
                return [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ]

            elif tool_name == "get_page_content":
                content = await self._browser.get_page_content(
                    url=tool_input["url"],
                    wait_for_js=tool_input.get("wait_for_js", True),
                )
                return {
                    "title": content.title,
                    "url": content.url,
                    "text": content.text_content,
                    "links_count": len(content.links),
                }

            elif tool_name == "store_finding":
                finding_type = tool_input.get("finding_type", "paraphrase")
                text_to_store = tool_input["text"]

                # Validate direct quotes against the source (blocking - agent needs result)
                if finding_type == "direct_quote":
                    validation = await validate_direct_quote(
                        quote=tool_input["text"],
                        source_url=tool_input["source_url"],
                        browser_tool=self._browser,
                    )

                    if not validation.valid:
                        return {
                            "status": "error",
                            "error": validation.error,
                            "finding_type": finding_type,
                            "suggestion": "Use 'paraphrase' or 'summary' instead, or copy the exact text from the source.",
                        }

                    # Use the validated text from the source
                    text_to_store = validation.matched_text
                    if validation.match_ratio < 1.0:
                        console.print(f"    [yellow]Quote corrected ({validation.match_ratio:.0%} match)[/yellow]")

                # Queue for async storage (non-blocking)
                self._finding_queue.enqueue(
                    StorageTask(
                        text=text_to_store,
                        source_url=tool_input["source_url"],
                        title=tool_input["title"],
                        relevance_notes=tool_input["relevance_notes"],
                        finding_type=finding_type,
                        author=tool_input.get("author"),
                        publication_date=tool_input.get("publication_date"),
                    )
                )
                return {"status": "queued", "finding_type": finding_type}

            elif tool_name == "search_findings":
                results = self._memory.search_findings(
                    query=tool_input["query"],
                    k=tool_input.get("k", 10),
                )
                return [
                    {
                        "text": f.text[:500] + "..." if len(f.text) > 500 else f.text,
                        "finding_type": f.finding_type,
                        "source": f.citation.source_url,
                        "title": f.citation.title,
                        "distance": round(score, 4),
                    }
                    for f, score in results
                ]

            elif tool_name == "get_memory_stats":
                return self._memory.get_statistics()

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": str(e)}

    async def _drain_queue(self) -> None:
        """Wait for all queued findings to be stored."""
        pending = self._finding_queue.pending_count
        if pending > 0:
            console.print(f"[dim]Waiting for {pending} findings to be stored...[/dim]")
        await self._finding_queue.stop()
        if self._finding_queue.failed_count > 0:
            console.print(
                f"[yellow]Warning: {self._finding_queue.failed_count} findings failed to store[/yellow]"
            )

    async def run(
        self,
        brief: ResearchBrief,
        wikipedia_context: WikipediaContext,
        research_plan: ResearchPlan,
    ) -> str:
        """
        Run the agent loop until completion.

        Returns:
            The final summary from the agent
        """
        # Start the finding queue worker
        await self._finding_queue.start()

        # Build system prompt
        system_prompt = build_system_prompt(brief, wikipedia_context, research_plan)

        # Initialize messages
        self._messages = [
            {
                "role": "user",
                "content": "Begin the research process according to the plan. Start with the first search queries.",
            }
        ]

        self._turn_count = 0
        max_turns = Config.MAX_AGENT_TURNS

        while self._turn_count < max_turns:
            # Check for shutdown request
            if self._shutdown_requested:
                console.print(Panel("[yellow]Shutdown requested. Saving findings...[/yellow]"))
                await self._drain_queue()
                return "Research interrupted by user. Partial findings have been saved."

            self._turn_count += 1

            console.print(f"\n[dim]Turn {self._turn_count}/{max_turns}[/dim]")

            # Call Claude
            api_params = {
                "model": self._model,
                "max_tokens": 16000 if self._thinking else 4096,
                "system": system_prompt,
                "tools": TOOLS,
                "messages": self._messages,
            }

            if self._thinking:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            response = self._client.messages.create(**api_params)

            # Track token usage
            self._total_input_tokens += response.usage.input_tokens
            self._total_output_tokens += response.usage.output_tokens

            # Parse response content - separate thinking from text
            text_content = ""
            thinking_content = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_content += block.thinking
                elif block.type == "text":
                    text_content += block.text

            # Show agent's thinking and reasoning if verbose
            if self._verbose:
                if thinking_content.strip():
                    # Truncate very long thinking for display
                    display_thinking = thinking_content.strip()
                    if len(display_thinking) > 1000:
                        display_thinking = display_thinking[:1000] + "\n... [truncated]"
                    console.print(Panel(
                        display_thinking,
                        title="[dim]Thinking[/dim]",
                        border_style="dim",
                    ))
                if text_content.strip():
                    console.print(Panel(
                        text_content.strip(),
                        title="[yellow]Agent[/yellow]",
                        border_style="dim",
                    ))

            if "RESEARCH_COMPLETE" in text_content:
                console.print(Panel("[green]Research complete![/green]"))
                await self._drain_queue()
                return text_content

            # Handle tool use
            if response.stop_reason == "tool_use":
                # Add assistant response to messages
                self._messages.append({"role": "assistant", "content": response.content})

                # Collect all tool calls
                tool_blocks = [block for block in response.content if block.type == "tool_use"]

                # Log what we're about to execute
                if len(tool_blocks) > 1:
                    console.print(f"  [cyan]Executing {len(tool_blocks)} tools in parallel...[/cyan]")

                for block in tool_blocks:
                    console.print(
                        f"  [cyan]Tool:[/cyan] {block.name}",
                        highlight=False,
                    )
                    # Log tool input for debugging
                    if block.name == "web_search":
                        console.print(
                            f"    [dim]Query: {block.input.get('query', '')}[/dim]"
                        )
                    elif block.name == "get_page_content":
                        url = block.input.get("url", "")
                        console.print(f"    [dim]URL: {url[:80]}...[/dim]" if len(url) > 80 else f"    [dim]URL: {url}[/dim]")
                    elif block.name == "store_finding":
                        finding_type = block.input.get("finding_type", "paraphrase")
                        text = block.input.get("text", "")[:80]
                        console.print(f"    [dim][{finding_type}] {text}...[/dim]")

                # Execute all tool calls in parallel
                async def execute_with_block(block):
                    result = await self._execute_tool(block.name, block.input)
                    return block, result

                results = await asyncio.gather(*[execute_with_block(block) for block in tool_blocks])

                # Process results and build tool_results list
                tool_results = []
                for block, result in results:
                    # Log result summary
                    if block.name == "web_search" and isinstance(result, list):
                        console.print(f"  [green]{block.name}: Found {len(result)} results[/green]")
                    elif block.name == "store_finding":
                        if isinstance(result, dict) and result.get("status") == "error":
                            console.print(f"  [red]{block.name}: {result.get('error', 'Error')}[/red]")
                        else:
                            console.print(f"  [green]{block.name}: Queued[/green]")
                    elif block.name == "get_page_content":
                        console.print(f"  [green]{block.name}: Done[/green]")
                    elif block.name == "get_memory_stats" and isinstance(result, dict):
                        console.print(
                            f"  [green]{block.name}: Findings: {result.get('total_findings', 0)}, Sources: {result.get('unique_sources', 0)}[/green]"
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

                # Add tool results
                self._messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                # Model finished without tool use
                self._messages.append({"role": "assistant", "content": response.content})

                # Check if it's done or needs prompting
                if text_content.strip():
                    # Check for completion
                    if "RESEARCH_COMPLETE" in text_content:
                        await self._drain_queue()
                        return text_content
                    # Otherwise, prompt to continue
                    self._messages.append(
                        {
                            "role": "user",
                            "content": "Please continue with the research. Use the tools to search and store findings.",
                        }
                    )
                else:
                    self._messages.append(
                        {
                            "role": "user",
                            "content": "Continue with the next search query from the research plan.",
                        }
                    )
            else:
                # Unexpected stop reason
                console.print(f"[yellow]Unexpected stop reason: {response.stop_reason}[/yellow]")
                self._messages.append({"role": "assistant", "content": response.content})
                self._messages.append(
                    {"role": "user", "content": "Please continue with the research."}
                )

        # Max turns reached
        console.print(
            Panel(f"[yellow]Maximum turns ({max_turns}) reached. Finalizing...[/yellow]")
        )
        await self._drain_queue()
        return "Research stopped due to maximum turns limit."

    @property
    def turn_count(self) -> int:
        """Return the number of turns taken."""
        return self._turn_count

    @property
    def total_input_tokens(self) -> int:
        """Return total input tokens used."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Return total output tokens used."""
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        """Return total tokens used."""
        return self._total_input_tokens + self._total_output_tokens
