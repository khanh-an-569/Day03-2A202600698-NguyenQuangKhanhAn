import os
import re
from typing import List, Dict, Any

from tavily import TavilyClient

def _tavily_web_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Call Tavily Search API with advanced search depth.
    Returns structured results with real URLs and snippets.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for item in response.get("results", [])[:max_results]:
        url = item.get("url")
        title = item.get("title") or ""
        snippet = item.get("content") or item.get("snippet") or ""

        if not url or not title:
            continue

        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        source = domain_match.group(1) if domain_match else "tavily"

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": source,
        })

    return results


# ── Public interface ──────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web and return structured results with real URLs and snippets.

    Strategy (tried in order):
      1. Tavily Search API with advanced depth         — requires TAVILY_API_KEY

    Each result dict has: title, url, snippet, source.
    """
    results = _tavily_web_search(query, max_results)
    if results:
        return results

    # Tavily unavailable
    return [{
        "title":   "Search unavailable",
        "url":     None,
        "snippet": (
            f"Could not retrieve live results for '{query}'. "
            "No TAVILY_API_KEY found or API unreachable. "
            "Use your training knowledge and mark all data as estimated."
        ),
        "source":  "tavily",
    }]