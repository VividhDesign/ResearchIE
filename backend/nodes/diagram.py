"""Diagram node — generates Mermaid diagrams for flagged sections."""
from __future__ import annotations
from langchain_core.messages import HumanMessage
from backend.state import ResearchState
from backend.utils.llm import get_fast_llm, extract_text
from backend.utils.prompts import DIAGRAM_PROMPT


def diagram_node(state: ResearchState) -> dict:
    """Generate Mermaid diagrams for sections that need visual content."""
    llm = get_fast_llm(temperature=0.2)

    # Find sections that need diagrams
    needs_diagram_titles = {
        s.title for s in state.plan.sections if s.needs_diagram
    } if state.plan else set()

    if not needs_diagram_titles:
        # Auto-generate diagram for first section
        if state.completed_sections:
            needs_diagram_titles = {state.completed_sections[0].title}

    updated_sections = []
    progress = []

    for section in state.completed_sections:
        if section.title in needs_diagram_titles:
            try:
                prompt = DIAGRAM_PROMPT.format(
                    section_title=section.title,
                    section_content=section.content[:2000],
                )
                response = llm.invoke([HumanMessage(content=prompt)])
                mermaid_code = extract_text(response.content).strip()

                # Clean up code fences if present
                if mermaid_code.startswith("```"):
                    lines = mermaid_code.split("\n")
                    mermaid_code = "\n".join(
                        lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                    )

                updated_sections.append(SectionResultWithDiagram(
                    title=section.title,
                    content=section.content,
                    word_count=section.word_count,
                    diagram_json={"mermaid": mermaid_code},
                ))
                progress.append(f"📊 Diagram generated for: '{section.title}'")
            except Exception as e:
                updated_sections.append(section)
                progress.append(f"Diagram generation failed for '{section.title}': {e}")
        else:
            updated_sections.append(section)

    return {
        "completed_sections": updated_sections,
        "status": "📊 Diagrams generated",
        "progress_log": progress or ["No diagrams needed"],
    }


# Import fix
from backend.state import SectionResult as SectionResultWithDiagram  # noqa
