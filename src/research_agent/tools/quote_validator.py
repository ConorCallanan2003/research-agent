"""Validation for direct quotes against source content."""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz


@dataclass
class QuoteValidationResult:
    """Result of validating a direct quote."""

    valid: bool
    matched_text: str | None = None  # The actual text from the source
    match_ratio: float = 0.0
    error: str | None = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison (collapse whitespace, lowercase)."""
    # Collapse all whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip and lowercase
    return text.strip().lower()


def find_exact_match(quote: str, source_text: str) -> str | None:
    """
    Find an exact match of the quote in the source text.

    Returns the matched text from the source (preserving original formatting)
    or None if not found.
    """
    normalized_quote = normalize_text(quote)
    normalized_source = normalize_text(source_text)

    if normalized_quote in normalized_source:
        # Find the position in normalized source
        start_idx = normalized_source.find(normalized_quote)

        # Map back to original source to preserve formatting
        # This is approximate - we find a window in the original text
        # that matches the normalized length
        original_words = source_text.split()
        normalized_words = normalized_quote.split()

        # Try to find the starting word
        for i in range(len(original_words) - len(normalized_words) + 1):
            window = ' '.join(original_words[i:i + len(normalized_words)])
            if normalize_text(window) == normalized_quote:
                return window

        # Fallback: return the quote as-is since we confirmed it exists
        return quote

    return None


def find_fuzzy_match(
    quote: str,
    source_text: str,
    min_ratio: float = 0.9
) -> tuple[str | None, float]:
    """
    Find a fuzzy match of the quote in the source text.

    Uses a sliding window approach to find the best matching substring.
    Uses RapidFuzz (C++ implementation) for fast fuzzy matching.

    Returns:
        Tuple of (matched_text, match_ratio) or (None, 0.0) if no match above threshold.
    """
    normalized_quote = normalize_text(quote)
    quote_words = normalized_quote.split()
    quote_word_count = len(quote_words)

    if quote_word_count == 0:
        return None, 0.0

    # Split source into words, preserving original text for extraction
    source_words_original = source_text.split()
    source_words_normalized = [w.lower() for w in source_words_original]

    best_match = None
    best_ratio = 0.0
    best_start = 0
    best_end = 0

    # Sliding window with varying sizes (allow some flexibility in length)
    min_window = max(1, quote_word_count - 5)
    max_window = quote_word_count + 5

    for window_size in range(min_window, min(max_window + 1, len(source_words_original) + 1)):
        for i in range(len(source_words_original) - window_size + 1):
            window_normalized = ' '.join(source_words_normalized[i:i + window_size])

            # Use RapidFuzz for fast similarity (returns 0-100, convert to 0-1)
            ratio = fuzz.ratio(normalized_quote, window_normalized) / 100.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = i + window_size

                # Early termination if we find a very good match
                if ratio >= 0.98:
                    best_match = ' '.join(source_words_original[best_start:best_end])
                    return best_match, best_ratio

    if best_ratio >= min_ratio:
        # Extract the original text (preserving formatting)
        best_match = ' '.join(source_words_original[best_start:best_end])
        return best_match, best_ratio

    return None, best_ratio


async def validate_direct_quote(
    quote: str,
    source_url: str,
    browser_tool,
    min_fuzzy_ratio: float = 0.9,
) -> QuoteValidationResult:
    """
    Validate a direct quote against its source.

    Args:
        quote: The quoted text to validate
        source_url: URL of the source to check against
        browser_tool: BrowserTool instance to fetch the page
        min_fuzzy_ratio: Minimum similarity ratio for fuzzy matching (default 0.9)

    Returns:
        QuoteValidationResult with validation status and matched text
    """
    # Fetch the source page
    try:
        page_content = await browser_tool.get_page_content(source_url)
        source_text = page_content.text_content
    except Exception as e:
        return QuoteValidationResult(
            valid=False,
            error=f"Failed to fetch source URL: {e}"
        )

    if not source_text or len(source_text.strip()) < 10:
        return QuoteValidationResult(
            valid=False,
            error="Source page has no readable content"
        )

    # Try exact match first
    exact_match = find_exact_match(quote, source_text)
    if exact_match:
        return QuoteValidationResult(
            valid=True,
            matched_text=exact_match,
            match_ratio=1.0,
        )

    # Try fuzzy match
    fuzzy_match, ratio = find_fuzzy_match(quote, source_text, min_fuzzy_ratio)
    if fuzzy_match:
        return QuoteValidationResult(
            valid=True,
            matched_text=fuzzy_match,
            match_ratio=ratio,
        )

    # No match found
    return QuoteValidationResult(
        valid=False,
        match_ratio=ratio,
        error=f"Quote not found in source. Best match had only {ratio:.0%} similarity. "
              f"Direct quotes must be exact or near-exact matches from the source text. "
              f"Consider using 'paraphrase' or 'summary' instead if you're restating the information."
    )
