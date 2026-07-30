"""Web Research node — uses Tavily to gather evidence from the web."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from backend.state import ResearchState, EvidenceItem

load_dotenv()


def web_research_node(state: ResearchState) -> dict:
    """Run Tavily searches and build the evidence pack."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    evidence = []
    queries = []

    # Use plan queries if available, else fall back to topic
    if state.plan and state.plan.search_queries:
        queries = state.plan.search_queries[:5]  # Max 5 queries
    else:
        queries = [state.topic]

    progress = []
    for query in queries:
        try:
            results = client.search(
                query=query,
                max_results=3,
                search_depth="advanced",
            )
            for r in results.get("results", []):
                evidence.append(EvidenceItem(
                    source=r.get("url", ""),
                    title=r.get("title", "Web Result"),
                    content=r.get("content", "")[:1500],  # Trim to 1500 chars
                    source_type="web",
                ))
            progress.append(f"Web search: '{query}' → {len(results.get('results', []))} results")
        except Exception as e:
            progress.append(f"Web search failed for '{query}': {e}")

    return {
        "evidence": evidence,
        "status": f"🌐 Web research complete: {len(evidence)} sources gathered",
        "progress_log": progress,
    }
