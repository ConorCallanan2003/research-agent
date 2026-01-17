"""Quick debug script for browser tool - no pytest needed."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_agent.tools.browser import BrowserTool


async def debug_duckduckgo():
    """Debug function to see what's happening with DuckDuckGo search."""
    print("Starting browser debug...")
    print("=" * 60)

    browser = BrowserTool(headless=False)  # Show the browser
    await browser.initialize()

    try:
        # Step 1: Go to DuckDuckGo homepage
        print("\n1. Navigating to DuckDuckGo homepage...")
        await browser._page.goto("https://duckduckgo.com")
        await asyncio.sleep(2)

        print(f"   Current URL: {browser._page.url}")

        # Take screenshot
        await browser._page.screenshot(path="debug_01_ddg_home.png")
        print("   Screenshot: debug_01_ddg_home.png")

        # Step 2: Try the search method
        print("\n2. Testing web_search method...")
        results = await browser.web_search("Python programming language", num_results=5)

        print(f"   Results returned: {len(results)}")

        # Take screenshot after search
        await browser._page.screenshot(path="debug_02_search_results.png")
        print("   Screenshot: debug_02_search_results.png")

        if results:
            print("\n   Results:")
            for i, r in enumerate(results, 1):
                print(f"   {i}. {r.title[:60]}")
                print(f"      URL: {r.url[:60]}")
        else:
            print("\n   No results returned!")
            print("   Checking page state...")
            print(f"   Current URL: {browser._page.url}")

            # Try to get some debug info from the page
            page_text = await browser._page.evaluate("() => document.body.innerText.substring(0, 500)")
            print(f"\n   Page text preview:\n   {page_text[:300]}...")

        # Step 3: Check what elements are on the page
        print("\n3. Checking for search result elements...")

        selectors_to_try = [
            ('[data-testid="result"]', "Result containers"),
            ('[data-testid="result-title-a"]', "Result title links"),
            ('[data-testid="result-snippet"]', "Result snippets"),
            ("article", "Article elements"),
        ]

        for selector, description in selectors_to_try:
            count = await browser._page.evaluate(f"() => document.querySelectorAll('{selector}').length")
            print(f"   {selector}: {count} elements ({description})")

    finally:
        await browser.close()
        print("\n" + "=" * 60)
        print("Debug complete. Check the screenshots in the current directory.")


if __name__ == "__main__":
    asyncio.run(debug_duckduckgo())
