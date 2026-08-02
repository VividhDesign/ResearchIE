"""Orchestrator (Planner) node — creates the full structured report plan."""
from __future__ import annotations
from langchain_core.messages import HumanMessage
from backend.state import ResearchState, ReportPlan
from backend.utils.llm import get_llm
from backend.utils.prompts import ORCHESTRATOR_PROMPT


def orchestrator_node(state: ResearchState) -> dict:
    """Generate a structured ReportPlan using the available evidence."""
    llm = get_llm(temperature=0.4, structured_output=ReportPlan)

    # Build evidence preview (first 300 chars of each item)
    evidence_preview = "\n".join([
        f"- [{e.source_type.upper()}] {e.title}: {e.content[:300]}..."
        for e in state.evidence[:10]
    ]) or "No evidence gathered yet (closed-book mode)"

    prompt = ORCHESTRATOR_PROMPT.format(
        topic=state.topic,
        audience=state.audience,
        tone=state.tone,
        route=state.route,
        evidence_preview=evidence_preview,
    )

    plan: ReportPlan = llm.invoke([HumanMessage(content=prompt)])

    critique_info = ""
    if state.critique and not state.critique.passed:
        critique_info = f"\n\nPREVIOUS CRITIQUE (attempt #{state.critique_count}):\n{state.critique.feedback}\nSections to improve: {', '.join(state.critique.sections_to_rewrite)}"
        # Add critique context to plan descriptions
        plan.executive_summary_prompt += critique_info

    return {
        "plan": plan,
        "status": f"📋 Plan created: '{plan.title}' with {len(plan.sections)} sections",
        "progress_log": [
            f"Orchestrator: planned {len(plan.sections)} sections",
            *[f"  → Section: '{s.title}'" for s in plan.sections],
        ],
    }
