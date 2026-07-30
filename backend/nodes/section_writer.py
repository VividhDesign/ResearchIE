"""Section Writer node — writes one section based on the plan and evidence."""
from __future__ import annotations
from langchain_core.messages import HumanMessage
from backend.state import ResearchState, SectionPlan, SectionResult
from backend.utils.llm import get_llm
from backend.utils.prompts import SECTION_WRITER_PROMPT


def section_writer_node(state: ResearchState, section: SectionPlan) -> dict:
    """Write a single report section. Called in parallel via Send()."""
    llm = get_llm(temperature=0.5)

    # Filter relevant evidence for this section
    relevant_evidence = []
    if section.requires_web_search:
        relevant_evidence.extend([e for e in state.evidence if e.source_type == "web"])
    if section.requires_rag:
        relevant_evidence.extend([e for e in state.evidence if e.source_type == "rag"])
    if not relevant_evidence:
        relevant_evidence = state.evidence[:5]  # fallback: use first 5

    # Format evidence pack
    evidence_text = "\n\n".join([
        f"**[{e.source_type.upper()}] {e.title}** ({e.source})\n{e.content}"
        for e in relevant_evidence[:6]  # Max 6 evidence items per section
    ]) or "No specific evidence — use your knowledge."

    prompt = SECTION_WRITER_PROMPT.format(
        report_title=state.plan.title,
        audience=state.plan.target_audience,
        tone=state.plan.tone,
        section_title=section.title,
        section_description=section.description,
        target_word_count=section.target_word_count,
        evidence=evidence_text,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    word_count = len(content.split())

    result = SectionResult(
        title=section.title,
        content=content,
        word_count=word_count,
    )

    return {
        "completed_sections": [result],
        "progress_log": [f"✍️ Section written: '{section.title}' ({word_count} words)"],
    }
