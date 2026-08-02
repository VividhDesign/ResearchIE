"""Main LangGraph StateGraph — wires all nodes together."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from backend.state import ResearchState
from backend.nodes.router import router_node
from backend.nodes.orchestrator import orchestrator_node
from backend.nodes.research import web_research_node
from backend.nodes.section_writer import section_writer_node
from backend.nodes.stitcher import stitcher_node
from backend.nodes.critic import critic_node
from backend.nodes.diagram import diagram_node
from backend.rag.retriever import rag_research_node

load_dotenv()

MAX_CRITIQUE_RETRIES = int(os.getenv("MAX_CRITIQUE_RETRIES", 2))


# ─── Conditional edge functions ───────────────────────────────────────────────

def decide_research_route(state: ResearchState):
    """After planning, decide which research nodes to run."""
    route = state.route
    if route == "hybrid":
        return "both"
    elif route == "rag":
        return "rag_only"
    elif route == "closed":
        return "skip"
    else:  # web or default
        return "web_only"


def fan_out_sections(state: ResearchState):
    """Create parallel Send() calls for each planned section."""
    if not state.plan or not state.plan.sections:
        return []
    return [
        Send("section_writer", {"state": state, "section": section})
        for section in state.plan.sections
    ]


def decide_after_critique(state: ResearchState) -> str:
    """After critique, decide to finalize or replan."""
    if not state.critique:
        return "finalize"
    if state.critique.passed or state.critique_count >= MAX_CRITIQUE_RETRIES:
        return "finalize"
    return "replan"


def decide_research_before_orchestrate(state: ResearchState) -> str:
    """After routing, decide whether to gather evidence first or go straight to orchestrator."""
    if state.route in ("web", "hybrid"):
        return "web_search"
    elif state.route == "rag":
        return "rag_search"
    else:
        return "orchestrate"


# ─── Section writer wrapper for Send() ────────────────────────────────────────

def section_writer_wrapper(inputs: dict) -> dict:
    """Wrapper so section_writer works with Send() fan-out."""
    state: ResearchState = inputs["state"]
    section = inputs["section"]
    return section_writer_node(state, section)


# ─── Finalize node ────────────────────────────────────────────────────────────

def finalize_node(state: ResearchState) -> dict:
    """Set the final report and generate PDF."""
    import re
    from backend.utils.pdf_export import export_to_pdf

    # Use stitched_report (may contain embedded base64 images from image_gen)
    final_report = state.stitched_report
    pdf_path = ""

    try:
        # Strip base64 data URIs before PDF generation (they crash ReportLab)
        pdf_safe_report = re.sub(r'!\[([^\]]*)\]\(data:[^)]+\)', r'[Image: \1]', final_report)
        # Also strip mermaid blocks (can't render in PDF)
        pdf_safe_report = re.sub(r'```mermaid.*?```', '', pdf_safe_report, flags=re.DOTALL)
        
        title = state.plan.title if state.plan else "Research Report"
        pdf_path = export_to_pdf(pdf_safe_report, title)
    except Exception as e:
        print(f"PDF export failed: {e}")

    return {
        "final_report": final_report,
        "pdf_path": pdf_path,
        "status": "✅ Report complete!",
        "progress_log": [f"Report finalized. PDF: {pdf_path}" if pdf_path else "Report finalized (PDF export failed)."],
    }


# ─── Pre-orchestrator research nodes ──────────────────────────────────────────

def pre_web_research(state: ResearchState) -> dict:
    """Initial web research before orchestration (generates queries from topic only)."""
    # Temporarily set plan to None for initial research using just the topic
    from tavily import TavilyClient
    import os
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    evidence = []
    from backend.state import EvidenceItem
    try:
        results = client.search(query=state.topic, max_results=5, search_depth="advanced")
        for r in results.get("results", []):
            evidence.append(EvidenceItem(
                source=r.get("url", ""),
                title=r.get("title", "Web Result"),
                content=r.get("content", "")[:1500],
                source_type="web",
            ))
    except Exception as e:
        pass
    return {
        "evidence": evidence,
        "status": "🌐 Initial web research complete",
        "progress_log": [f"Pre-research: {len(evidence)} initial sources gathered"],
    }


def pre_rag_research(state: ResearchState) -> dict:
    """Initial RAG research before orchestration."""
    return rag_research_node(state)


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the full research agent graph."""
    builder = StateGraph(ResearchState)

    # Add all nodes
    builder.add_node("router", router_node)
    builder.add_node("pre_web_research", pre_web_research)
    builder.add_node("pre_rag_research", pre_rag_research)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("web_research", web_research_node)
    builder.add_node("rag_research", rag_research_node)
    builder.add_node("section_writer", section_writer_wrapper)
    builder.add_node("stitcher", stitcher_node)
    builder.add_node("critic", critic_node)
    builder.add_node("diagram", diagram_node)
    builder.add_node("finalize", finalize_node)

    # Entry point
    builder.set_entry_point("router")

    # Router → initial research or orchestrator
    builder.add_conditional_edges(
        "router",
        decide_research_before_orchestrate,
        {
            "web_search": "pre_web_research",
            "rag_search": "pre_rag_research",
            "orchestrate": "orchestrator",
        }
    )

    # Pre-research → orchestrator
    builder.add_edge("pre_web_research", "orchestrator")
    builder.add_edge("pre_rag_research", "orchestrator")

    # Orchestrator → deep research (now with plan's specific queries)
    builder.add_conditional_edges(
        "orchestrator",
        decide_research_route,
        {
            "both": "web_research",
            "web_only": "web_research",
            "rag_only": "rag_research",
            "skip": "section_writer",  # fan-out directly
        }
    )

    # Research nodes → fan-out to section writers
    builder.add_conditional_edges("web_research", fan_out_sections, ["section_writer"])
    builder.add_conditional_edges("rag_research", fan_out_sections, ["section_writer"])

    # Section writers → stitcher (fan-in happens automatically via Annotated list)
    builder.add_edge("section_writer", "stitcher")

    # Stitcher → critic
    builder.add_edge("stitcher", "critic")

    # Critic → finalize or replan loop
    builder.add_conditional_edges(
        "critic",
        decide_after_critique,
        {
            "finalize": "diagram",
            "replan": "orchestrator",
        }
    )

    # Diagram → Finalize
    builder.add_edge("diagram", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Singleton graph instance
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
