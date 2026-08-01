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
    stitched = state.stitched_report

    # Deduplicate sections so we don't generate diagrams multiple times for the same section
    unique_sections = {}
    for s in state.completed_sections:
        unique_sections[s.title] = s

    for section in unique_sections.values():
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
                
                # Inject the diagram directly into the stitched report under the relevant heading
                if stitched:
                    # Try to insert after the heading
                    heading_str = f"## {section.title}"
                    if heading_str in stitched:
                        stitched = stitched.replace(
                            heading_str, 
                            f"{heading_str}\n\n```mermaid\n{mermaid_code}\n```\n"
                        )
                    else:
                        # Fallback: append at end
                        stitched += f"\n\n```mermaid\n{mermaid_code}\n```\n"
                        
                progress.append(f"📊 Diagram generated for: '{section.title}'")
            except Exception as e:
                updated_sections.append(section)
                error_msg = f"Diagram generation failed for '{section.title}' (Rate limit or API error)"
                progress.append(error_msg)
                
                # Append error into the report so the user sees WHY the diagram is missing
                if stitched:
                    heading_str = f"## {section.title}"
                    if heading_str in stitched:
                        stitched = stitched.replace(
                            heading_str, 
                            f"{heading_str}\n\n> ⚠️ **{error_msg}**\n\n"
                        )
        else:
            updated_sections.append(section)

    return {
        "completed_sections": updated_sections,
        "stitched_report": stitched,
        "status": "📊 Diagrams generated",
        "progress_log": progress or ["No diagrams needed"],
    }


# Import fix
from backend.state import SectionResult as SectionResultWithDiagram  # noqa
