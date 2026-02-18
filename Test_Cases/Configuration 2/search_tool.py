from duckduckgo_search import DDGS
from typing import List, Dict


def perform_web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Executes a DuckDuckGo search and returns a list of results.
    Used by proofreader to verify facts.
    """
    print(f"Web Search: Searching for '{query}'...")
    results = []
    try:
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, max_results=max_results)
            for r in ddgs_gen:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", ""),
                    }
                )
    except Exception as e:
        print(f"⚠️ Web Search Error: {e}")
        return []

    return results
