"""
Web search utilities using DuckDuckGo (free, no API key required).

Provides topic-specific search and page content fetching for grounding
quiz questions in current official documentation.
"""
from __future__ import annotations

import re
import time
from typing import Any

import requests

from config.settings import (
    SEARCH_NUM_RESULTS,
    SEARCH_FETCH_TOP_N,
    SEARCH_PAGE_MAX_CHARS,
    SEARCH_DELAY_SECONDS,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def search_topic(
    exam_title: str,
    topic_name: str,
    num_results: int = SEARCH_NUM_RESULTS,
) -> list[dict[str, str]]:
    """
    Search DuckDuckGo for official documentation related to a topic.

    Args:
        exam_title: Name of the certification exam (e.g., "GCP PMLE").
        topic_name: The specific topic to search for.
        num_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, body (snippet).
    """
    from ddgs import DDGS

    # Bias toward official GCP documentation
    query = (
        f"{exam_title} {topic_name} official documentation "
        f"site:cloud.google.com OR site:developers.google.com"
    )

    results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                })
    except Exception:
        # Fallback: broader search without site restriction
        try:
            fallback_query = f"{exam_title} {topic_name} official documentation"
            with DDGS() as ddgs:
                for r in ddgs.text(fallback_query, max_results=num_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": r.get("body", ""),
                    })
        except Exception:
            pass

    return results


def fetch_page_content(url: str, max_chars: int = SEARCH_PAGE_MAX_CHARS) -> str:
    """
    Fetch a webpage and return its plain-text content (HTML tags stripped).

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return.

    Returns:
        Plain-text content, or empty string on failure.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        resp.raise_for_status()
        html = resp.text

        # Remove script and style blocks entirely
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)

        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars]
    except Exception:
        return ""


def gather_source_content(
    exam_title: str,
    topic: dict[str, Any],
    delay: float = SEARCH_DELAY_SECONDS,
) -> tuple[str, list[str]]:
    """
    Search for official documentation for a topic, fetch page content,
    and return combined source text with verified URLs.

    Args:
        exam_title: Exam name for search context.
        topic: Topic dict with keys: name, description, subtopics.
        delay: Seconds to wait after search (DuckDuckGo rate limiting).

    Returns:
        Tuple of (combined_source_text, [verified_urls]).
    """
    topic_name = topic.get("name", "")
    search_results = search_topic(exam_title, topic_name)
    time.sleep(delay)  # respect DuckDuckGo rate limits

    fetched_parts: list[str] = []
    source_urls: list[str] = []

    for result in search_results[:SEARCH_FETCH_TOP_N]:
        url = result.get("url", "")
        if not url:
            continue

        content = fetch_page_content(url)
        if content:
            fetched_parts.append(
                f"[Title: {result.get('title', url)}]\n"
                f"[URL: {url}]\n"
                f"{content}"
            )
            source_urls.append(url)

    if not fetched_parts:
        # Use snippets from search results as fallback
        for result in search_results:
            snippet = result.get("body", "")
            url = result.get("url", "")
            if snippet:
                fetched_parts.append(
                    f"[Title: {result.get('title', '')}]\n"
                    f"[URL: {url}]\n"
                    f"{snippet}"
                )
                if url:
                    source_urls.append(url)

    combined = "\n\n===\n\n".join(fetched_parts)
    return combined, source_urls
