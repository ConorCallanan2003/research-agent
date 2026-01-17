"""Playwright-based browser tool for web research."""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote_plus

from playwright.async_api import Browser, Page, async_playwright

from research_agent.config import Config


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str


@dataclass
class PageContent:
    """Extracted content from a web page."""

    url: str
    title: str
    text_content: str
    links: list[dict[str, str]]  # [{"text": "...", "href": "..."}]
    extraction_timestamp: str


class BrowserTool:
    """Playwright-based browser tool for web research with parallel page support."""

    def __init__(self, headless: bool | None = None, max_concurrent_pages: int = 4):
        self._browser: Browser | None = None
        self._playwright = None
        self._headless = headless if headless is not None else Config.BROWSER_HEADLESS
        self._max_concurrent = max_concurrent_pages
        self._semaphore: asyncio.Semaphore | None = None

    async def initialize(self) -> None:
        """Launch browser instance."""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def _create_page(self) -> Page:
        """Create a new page with standard configuration."""
        page = await self._browser.new_page()
        await page.set_extra_http_headers(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        return page

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self) -> "BrowserTool":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def web_search(
        self, query: str, num_results: int = 10
    ) -> list[SearchResult]:
        """
        Execute web search using DuckDuckGo and return results.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if not self._browser:
            await self.initialize()

        async with self._semaphore:
            page = await self._create_page()
            try:
                # Build DuckDuckGo search URL
                encoded_query = quote_plus(query)
                search_url = f"https://duckduckgo.com/?q={encoded_query}&t=h_&ia=web"

                await page.goto(search_url, timeout=Config.PAGE_TIMEOUT_MS)

                # Wait for search results to load
                await page.wait_for_selector('[data-testid="result"]', timeout=10000)

                # Give the page a moment to fully render
                await asyncio.sleep(0.5)

                # Extract search results from DuckDuckGo
                results = await page.evaluate(
                    """
                    () => {
                        const results = [];
                        const seen = new Set();

                        // DuckDuckGo result items
                        const items = document.querySelectorAll('[data-testid="result"]');

                        for (const item of items) {
                            // Get the title link
                            const titleLink = item.querySelector('a[data-testid="result-title-a"]');
                            const snippetEl = item.querySelector('[data-testid="result-snippet"]');

                            if (titleLink) {
                                const url = titleLink.href;

                                // Skip duplicates and DuckDuckGo internal links
                                if (!url || url.includes('duckduckgo.com') || seen.has(url)) continue;
                                seen.add(url);

                                results.push({
                                    title: titleLink.textContent || '',
                                    url: url,
                                    snippet: snippetEl ? snippetEl.textContent || '' : ''
                                });
                            }
                        }

                        return results;
                    }
                """
                )

                return [
                    SearchResult(
                        title=r["title"], url=r["url"], snippet=r["snippet"]
                    )
                    for r in results[:num_results]
                ]

            except Exception as e:
                # Return empty results on error
                return []
            finally:
                await page.close()

    async def get_page_content(
        self,
        url: str,
        wait_for_js: bool = True,
        timeout_ms: int | None = None,
    ) -> PageContent:
        """
        Navigate to URL and extract page content.

        Args:
            url: Full URL to visit
            wait_for_js: Wait for JavaScript to render
            timeout_ms: Custom timeout in milliseconds

        Returns:
            PageContent with extracted text and metadata
        """
        if not self._browser:
            await self.initialize()

        timeout = timeout_ms or Config.PAGE_TIMEOUT_MS

        async with self._semaphore:
            page = await self._create_page()
            try:
                await page.goto(url, timeout=timeout)

                if wait_for_js:
                    # Wait for content to stabilize
                    await page.wait_for_load_state("networkidle", timeout=timeout)

                # Extract title
                title = await page.title()

                # Extract main text content
                text_content = await page.evaluate(
                    """
                    () => {
                        // Remove script and style elements
                        const scripts = document.querySelectorAll('script, style, noscript, nav, footer, header');
                        scripts.forEach(el => el.remove());

                        // Try to find main content area
                        const mainSelectors = ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.article'];
                        let mainContent = null;

                        for (const selector of mainSelectors) {
                            mainContent = document.querySelector(selector);
                            if (mainContent) break;
                        }

                        // Fall back to body if no main content found
                        const target = mainContent || document.body;

                        // Get text content and clean it up
                        let text = target.innerText || target.textContent || '';

                        // Clean up whitespace
                        text = text.replace(/\\s+/g, ' ').trim();

                        return text;
                    }
                """
                )

                # Truncate if too long
                if len(text_content) > Config.MAX_CONTENT_LENGTH:
                    text_content = text_content[: Config.MAX_CONTENT_LENGTH] + "... [truncated]"

                # Extract links
                links = await page.evaluate(
                    """
                    () => {
                        const links = [];
                        const anchors = document.querySelectorAll('a[href^="http"]');

                        for (const a of anchors) {
                            if (a.textContent && a.textContent.trim()) {
                                links.push({
                                    text: a.textContent.trim().substring(0, 100),
                                    href: a.href
                                });
                            }
                        }

                        return links.slice(0, 50);  // Limit to 50 links
                    }
                """
                )

                return PageContent(
                    url=url,
                    title=title,
                    text_content=text_content,
                    links=links,
                    extraction_timestamp=datetime.utcnow().isoformat(),
                )

            except Exception as e:
                return PageContent(
                    url=url,
                    title="Error loading page",
                    text_content=f"Failed to load page: {str(e)}",
                    links=[],
                    extraction_timestamp=datetime.utcnow().isoformat(),
                )
            finally:
                await page.close()
