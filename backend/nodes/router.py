"""Router node — classifies the topic and decides research strategy."""
from __future__ import annotations
from langchain_core.messages import HumanMessage
from backend.state import ResearchState
from backend.utils.llm import get_fast_llm, extract_text
from backend.utils.prompts import ROUTER_PROMPT


def router_node(state: ResearchState) -> dict:
    """Classify the topic and set the research route."""
    llm = get_fast_llm(temperature=0.0)

    prompt = ROUTER_PROMPT.format(
        topic=state.topic,
        has_docs=bool(state.uploaded_doc_paths),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    route_text = extract_text(response.content).strip().lower()

    # Validate route
    valid_routes = {"closed", "web", "rag", "hybrid"}
    route = route_text if route_text in valid_routes else "web"

    # If no docs uploaded, never use rag-only
    if not state.uploaded_doc_paths and route in ("rag", "hybrid"):
        route = "web"

    return {
        "route": route,
        "status": f"🔀 Route decided: {route.upper()}",
        "progress_log": [f"Router: topic classified as '{route}'"],
    }
