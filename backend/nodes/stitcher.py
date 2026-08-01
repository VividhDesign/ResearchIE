"""Stitcher node — merges all sections into a cohesive report."""
from __future__ import annotations
from langchain_core.messages import HumanMessage
from backend.state import ResearchState
from backend.utils.llm import get_llm, extract_text
from backend.utils.prompts import STITCHER_PROMPT


def stitcher_node(state: ResearchState) -> dict:
    """Merge all completed sections and generate the stitched report."""
    llm = get_llm(temperature=0.3)

    if not state.completed_sections:
        return {
            "stitched_report": "# Error\nNo sections were generated.",
            "progress_log": ["Stitcher: No sections to merge!"],
        }

    # Order sections according to the plan
    plan_order = {s.title: i for i, s in enumerate(state.plan.sections)}
    sorted_sections = sorted(
        state.completed_sections,
        key=lambda s: plan_order.get(s.title, 999)
    )

    # Format all sections for stitching prompt
    sections_content = "\n\n---\n\n".join([
        f"## {s.title}\n\n{s.content}"
        for s in sorted_sections
    ])

    prompt = STITCHER_PROMPT.format(
        title=state.plan.title,
        exec_summary_prompt=state.plan.executive_summary_prompt,
        audience=state.plan.target_audience,
        tone=state.plan.tone,
        sections_content=sections_content,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    framing = extract_text(response.content)

    # Programmatically assemble the final report
    # The LLM outputs Executive Summary, Key Takeaways, References.
    # We will insert the main body between Executive Summary and Key Takeaways.
    
    parts = framing.split("# Key Takeaways")
    if len(parts) == 2:
        top_half = parts[0].strip()
        bottom_half = "\n\n# Key Takeaways\n" + parts[1].strip()
    else:
        # Fallback if the LLM didn't format exactly as requested
        top_half = framing
        bottom_half = ""

    # Check if a diagram was added in diagram.py (which runs after stitcher).
    # Wait, diagram.py actually runs *after* stitcher in the graph? Wait! Let's check graph.py in a sec.
    # We just stitch the sections.
    body = ""
    for s in sorted_sections:
        body += f"\n\n## {s.title}\n\n{s.content}\n"
        if s.diagram_json and "mermaid" in s.diagram_json:
            body += f"\n```mermaid\n{s.diagram_json['mermaid']}\n```\n"

    final_report = f"# {state.plan.title}\n\n{top_half}\n{body}\n{bottom_half}"

    total_words = sum(s.word_count for s in sorted_sections)

    return {
        "stitched_report": final_report,
        "status": f"📎 Report stitched: ~{total_words} words across {len(sorted_sections)} sections",
        "progress_log": [f"Stitcher: merged {len(sorted_sections)} sections (~{total_words} words)"],
    }
