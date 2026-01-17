"""Tests for the browser tool."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBrowserTool:
    """Tests for BrowserTool."""

    async def test_web_search_returns_results(self):
        """Test that web search returns results for a simple query."""
        from research_agent.tools.browser import BrowserTool

        async with BrowserTool(headless=True) as browser:
            results = await browser.web_search("Python programming", num_results=5)

            print(f"\nSearch results count: {len(results)}")
            for i, r in enumerate(results):
                print(f"  {i+1}. {r.title[:50]}... - {r.url[:50]}...")

            assert len(results) > 0, "Expected at least one search result"
            assert results[0].title, "Expected result to have a title"
            assert results[0].url, "Expected result to have a URL"

    async def test_get_page_content(self):
        """Test that we can fetch page content."""
        from research_agent.tools.browser import BrowserTool

        async with BrowserTool(headless=True) as browser:
            content = await browser.get_page_content("https://example.com")

            print(f"\nPage title: {content.title}")
            print(f"Content length: {len(content.text_content)}")
            print(f"Links count: {len(content.links)}")

            assert content.title, "Expected page to have a title"
            assert len(content.text_content) > 0, "Expected page to have content"


# Quick manual test
if __name__ == "__main__":
    async def run_tests():
        tests = TestBrowserTool()

        print("Running test_web_search_returns_results...")
        try:
            await tests.test_web_search_returns_results()
            print("PASSED\n")
        except Exception as e:
            print(f"FAILED: {e}\n")

        print("Running test_get_page_content...")
        try:
            await tests.test_get_page_content()
            print("PASSED\n")
        except Exception as e:
            print(f"FAILED: {e}\n")

    asyncio.run(run_tests())
