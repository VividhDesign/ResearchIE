"""Critic node — quality judges the stitched report and triggers rewrites."""
from __future__ import annotations
from langchain_core.messages import HumanMessage
from backend.state import ResearchState, CritiqueResult
from backend.utils.llm import get_llm
from backend.utils.prompts import CRITIC_PROMPT


def critic_node(state: ResearchState) -> dict:
    """Evaluate report quality and decide if rewrite is needed."""
    llm = get_llm(temperature=0.1, structured_output=CritiqueResult)

    prompt = CRITIC_PROMPT.format(
        title=state.plan.title,
        report_content=state.stitched_report[:8000],  # Truncate for context window
    )

    critique: CritiqueResult = llm.invoke([HumanMessage(content=prompt)])
    new_count = state.critique_count + 1

    status_emoji = "✅" if critique.passed else "🔄"
    status_msg = (
        f"{status_emoji} Critic: Score {critique.overall_score}/10 — {'PASSED' if critique.passed else 'NEEDS REWRITE'}"
    )

    progress = [
        f"Critic pass #{new_count}: score={critique.overall_score}/10, passed={critique.passed}",
    ]
    if not critique.passed:
        progress.append(f"  Issues: {'; '.join(critique.issues[:3])}")
        progress.append(f"  Sections to rewrite: {', '.join(critique.sections_to_rewrite)}")

    return {
        "critique": critique,
        "critique_count": new_count,
        "status": status_msg,
        "progress_log": progress,
    }
