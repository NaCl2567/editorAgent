
from typing import Annotated, Literal

from duckduckgo_search import DDGS

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    if not query.strip():
        return ""

    formatted_results = []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        for idx, item in enumerate(raw_results, start=1):
            title = item.get("title", "").strip()
            href = item.get("href", "").strip()
            body = item.get("body", "").strip()

            formatted_results.append(
                f"[{idx}]\n"
                f"Title: {title}\n"
                f"URL: {href}\n"
                f"Summary: {body}"
            )

    except Exception as e:
        return (
            "[DuckDuckGo search failed]\n"
            f"Query: {query}\n"
            f"Error: {e}"
        )

    if not formatted_results:
        return (
            "[DuckDuckGo search returned no results]\n"
            f"Query: {query}"
        )

    return "\n\n".join(formatted_results)

from tavily import TavilyClient
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
import httpx

tavily_client = TavilyClient()

def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


def tavily_search(
    query: str,
    max_results: int = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    # Use Tavily to discover URLs
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
    )

    # Fetch full content for each URL
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]

        # Fetch webpage content
        content = fetch_webpage_content(url)

        result_text = f"""## {title}
        **URL:** {url}

        {content}

        ---
        """
        result_texts.append(result_text)

    # Format final response
    response = f"""🔍 Found {len(result_texts)} result(s) for '{query}':

                {chr(10).join(result_texts)}"""

    return response