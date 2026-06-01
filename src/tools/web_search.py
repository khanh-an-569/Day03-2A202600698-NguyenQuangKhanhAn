import html
import re
from typing import List, Dict, Any
from urllib.parse import parse_qs, unquote, urlparse

import requests


DDG_URL = "https://html.duckduckgo.com/html/"


def _clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path == "/l/":
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
    return url


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Lightweight DuckDuckGo HTML search.

    Returns a list of structured search results that can be fed back into a ReAct agent.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.post(
            DDG_URL,
            data={"q": query},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
    except Exception as exc:
        return [
            {
                "title": "Search failed",
                "url": None,
                "snippet": f"Unable to query DuckDuckGo: {exc}",
                "source": "duckduckgo",
            }
        ]

    html_text = response.text
    blocks = re.findall(r'<div class="result__body[^>]*>(.*?)</div>\s*</div>', html_text, re.S)
    results: List[Dict[str, Any]] = []

    for block in blocks:
        if len(results) >= max_results:
            break

        title_match = re.search(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        snippet_match = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block, re.S)

        if not title_match:
            continue

        url = _normalize_url(html.unescape(title_match.group(1)))
        title = _clean_text(title_match.group(2))
        snippet = _clean_text(snippet_match.group(1)) if snippet_match else ""

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": "duckduckgo",
            }
        )

    if not results:
        results.append(
            {
                "title": "No results parsed",
                "url": None,
                "snippet": f"DuckDuckGo returned content for query: {query}, but no results were parsed.",
                "source": "duckduckgo",
            }
        )

    return results