"""Async queue for background finding storage."""

import asyncio
from dataclasses import dataclass

from rich.console import Console

from research_agent.tools.memory import MemoryTool

console = Console()


@dataclass
class StorageTask:
    """A finding waiting to be stored."""

    text: str
    source_url: str
    title: str
    relevance_notes: str
    finding_type: str
    author: str | None = None
    publication_date: str | None = None


class FindingQueue:
    """
    Async queue for storing findings in the background.

    Validation (e.g., direct quote checking) happens synchronously before queueing.
    The actual storage (embedding generation + write) happens asynchronously.
    """

    def __init__(self, memory_tool: MemoryTool):
        self._memory = memory_tool
        self._queue: asyncio.Queue[StorageTask | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._stored_count = 0
        self._failed_count = 0
        self._errors: list[str] = []

    async def start(self) -> None:
        """Start the background worker."""
        self._worker_task = asyncio.create_task(self._worker())

    async def drain(self) -> None:
        """Wait for all queued items to be processed."""
        await self._queue.join()

    async def stop(self) -> None:
        """Stop worker after draining the queue."""
        await self.drain()
        # Send sentinel to stop the worker gracefully
        await self._queue.put(None)
        if self._worker_task:
            await self._worker_task

    def enqueue(self, task: StorageTask) -> None:
        """Add a finding to the storage queue (non-blocking)."""
        self._queue.put_nowait(task)

    @property
    def stored_count(self) -> int:
        """Number of findings successfully stored."""
        return self._stored_count

    @property
    def failed_count(self) -> int:
        """Number of findings that failed to store."""
        return self._failed_count

    @property
    def pending_count(self) -> int:
        """Number of findings waiting in queue."""
        return self._queue.qsize()

    @property
    def errors(self) -> list[str]:
        """List of error messages from failed storage attempts."""
        return self._errors.copy()

    async def _worker(self) -> None:
        """Background worker that processes storage tasks."""
        while True:
            try:
                task = await self._queue.get()

                if task is None:
                    self._queue.task_done()
                    break

                try:
                    # Run blocking storage in thread pool
                    await asyncio.to_thread(
                        self._memory.store_finding_from_dict,
                        text=task.text,
                        source_url=task.source_url,
                        title=task.title,
                        relevance_notes=task.relevance_notes,
                        finding_type=task.finding_type,
                        author=task.author,
                        publication_date=task.publication_date,
                    )
                    self._stored_count += 1
                except Exception as e:
                    self._failed_count += 1
                    self._errors.append(f"Failed to store finding: {e}")
                    console.print(f"  [red]Queue error: {e}[/red]")
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
