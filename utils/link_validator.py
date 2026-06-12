"""
HTTP link validation utility.

Validates URLs concurrently using a thread pool to avoid blocking the
Streamlit main thread for long periods.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config.settings import LINK_VALIDATE_TIMEOUT, LINK_VALIDATE_MAX_WORKERS

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def validate_link(url: str, timeout: int = LINK_VALIDATE_TIMEOUT) -> tuple[bool, int]:
    """
    Check whether a single URL is reachable.

    Returns:
        (is_valid, http_status_code). status_code is 0 on connection error.
    """
    if not url or not url.startswith("http"):
        return False, 0
    try:
        resp = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers=_HEADERS,
        )
        return resp.status_code < 400, resp.status_code
    except requests.RequestException:
        # Try GET as fallback (some servers reject HEAD)
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers=_HEADERS,
                stream=True,
            )
            resp.close()
            return resp.status_code < 400, resp.status_code
        except requests.RequestException:
            return False, 0


def validate_links_batch(
    urls: list[str],
    timeout: int = LINK_VALIDATE_TIMEOUT,
    max_workers: int = LINK_VALIDATE_MAX_WORKERS,
) -> dict[str, tuple[bool, int]]:
    """
    Validate multiple URLs concurrently.

    Args:
        urls: List of URLs to validate.
        timeout: Per-request timeout in seconds.
        max_workers: Maximum concurrent validation threads.

    Returns:
        Dict mapping url -> (is_valid, http_status_code).
    """
    unique_urls = list(dict.fromkeys(u for u in urls if u))  # deduplicate, preserve order
    results: dict[str, tuple[bool, int]] = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_urls) or 1)) as executor:
        future_to_url = {
            executor.submit(validate_link, url, timeout): url
            for url in unique_urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = (False, 0)

    return results
