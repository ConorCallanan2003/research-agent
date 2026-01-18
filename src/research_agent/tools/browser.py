"""Playwright-based browser tool for web research."""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote_plus

from playwright.async_api import Browser, Page, async_playwright
from playwright_stealth.stealth import Stealth

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
        self._page_cache: dict[str, PageContent] = {}  # URL -> cached content
        self._url_locks: dict[str, asyncio.Lock] = {}  # Per-URL locks to prevent duplicate fetches

    async def initialize(self) -> None:
        """Launch browser instance."""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def _create_page(self) -> Page:
        """Create a new page with standard configuration and stealth techniques."""
        page = await self._browser.new_page()

        # Set realistic viewport
        await page.set_viewport_size({"width": 1920, "height": 1080})

        # Set comprehensive headers
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        })

        # Apply comprehensive stealth techniques using playwright-stealth
        stealth_config = Stealth(
            navigator_platform_override="MacIntel",  # Match our user agent
            navigator_user_agent_override="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        await stealth_config.apply_stealth_async(page)

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
        Execute web search using Yahoo and return results.

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
                # Build Yahoo search URL
                encoded_query = quote_plus(query)
                search_url = f"https://search.yahoo.com/search?p={encoded_query}"

                await page.goto(search_url, timeout=Config.PAGE_TIMEOUT_MS)

                # Handle Yahoo's privacy consent dialog if it appears
                try:
                    accept_button = page.locator('button:has-text("Accept all")')
                    if await accept_button.is_visible(timeout=2000):
                        await accept_button.click()
                        await asyncio.sleep(1)
                except Exception:
                    # No consent dialog or already accepted
                    pass

                # Wait for search results to load
                await page.wait_for_selector('.algo', timeout=10000)

                # Give the page a moment to fully render
                await asyncio.sleep(0.5)

                # Extract search results from Yahoo
                results = await page.evaluate(
                    """
                    () => {
                        const results = [];
                        const seen = new Set();

                        // Yahoo result items
                        const items = document.querySelectorAll('.algo');

                        for (const item of items) {
                            // Get the title link - try multiple selectors
                            let titleLink = item.querySelector('h3 a');
                            if (!titleLink) titleLink = item.querySelector('.title a');
                            if (!titleLink) titleLink = item.querySelector('a');

                            const snippetEl = item.querySelector('.compText, .lh-16, p');

                            if (titleLink) {
                                const url = titleLink.href;

                                // Skip duplicates and Yahoo internal links
                                if (!url || url.includes('yahoo.com/search') || seen.has(url)) continue;
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
        # Check cache first (fast path, no lock needed)
        if url in self._page_cache:
            return self._page_cache[url]

        # Get or create a lock for this URL to prevent duplicate fetches
        if url not in self._url_locks:
            self._url_locks[url] = asyncio.Lock()

        async with self._url_locks[url]:
            # Check cache again (another request may have populated it while we waited)
            if url in self._page_cache:
                return self._page_cache[url]

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

                    result = PageContent(
                        url=url,
                        title=title,
                        text_content=text_content,
                        links=links,
                        extraction_timestamp=datetime.utcnow().isoformat(),
                    )
                    # Cache successful fetches
                    self._page_cache[url] = result
                    return result

                except Exception as e:
                    # Don't cache errors - allow retry
                    return PageContent(
                        url=url,
                        title="Error loading page",
                        text_content=f"Failed to load page: {str(e)}",
                        links=[],
                        extraction_timestamp=datetime.utcnow().isoformat(),
                    )
                finally:
                    await page.close()
