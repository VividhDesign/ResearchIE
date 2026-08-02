from __future__ import annotations
from typing import Annotated, Any, Optional, List
from pydantic import BaseModel, Field
import operator


# ─── Plan schemas (Orchestrator output) ──────────────────────────────────────

class SectionPlan(BaseModel):
    """A single planned section of the report."""
    title: str = Field(description="Section title, e.g. 'Architectural Overview'")
    description: str = Field(description="What this section should cover (2-3 sentences)")
    requires_web_search: bool = Field(default=False, description="Does this section need fresh web data?")
    requires_rag: bool = Field(default=False, description="Does this section need private document knowledge?")
    target_word_count: int = Field(default=400, description="Approximate word count for this section")
    needs_diagram: bool = Field(default=False, description="Should a Mermaid diagram be generated for this section?")


class ReportPlan(BaseModel):
    """The full structured plan produced by the Orchestrator."""
    title: str = Field(description="Final report title")
    executive_summary_prompt: str = Field(description="Brief instructions for the executive summary")
    target_audience: str = Field(description="Who is this report for (e.g. 'ML engineers', 'startup founders')")
    tone: str = Field(description="Writing tone: 'technical', 'analytical', 'executive', 'academic'")
    sections: list[SectionPlan] = Field(description="Ordered list of sections to write")
    search_queries: list[str] = Field(default_factory=list, description="Web search queries to run for research")
    rag_queries: list[str] = Field(default_factory=list, description="Queries to run against uploaded documents")


# ─── Evidence schemas ─────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of research evidence."""
    source: str  # URL or document name
    title: str
    content: str
    source_type: str  # 'web' or 'rag'


# ─── Section result ───────────────────────────────────────────────────────────

class SectionResult(BaseModel):
    """A completed section."""
    title: str
    content: str  # Full markdown content
    word_count: int
    diagram_json: Optional[dict] = None  # Mermaid diagram spec if needed


# ─── Critique result ──────────────────────────────────────────────────────────

class CritiqueResult(BaseModel):
    """Structured feedback from the Critic agent."""
    passed: bool = Field(description="True if the report passes quality bar")
    overall_score: int = Field(description="Score 1-10")
    issues: list[str] = Field(default_factory=list, description="List of specific issues found")
    sections_to_rewrite: list[str] = Field(default_factory=list, description="Titles of sections needing rewrite")
    feedback: str = Field(description="Overall actionable feedback for rewriting")


# ─── LangGraph State ──────────────────────────────────────────────────────────

class ResearchState(BaseModel):
    """The shared state passed through all nodes of the LangGraph."""

    # Input
    topic: str = ""
    audience: str = "general technical audience"
    tone: str = "analytical"
    uploaded_doc_paths: list[str] = Field(default_factory=list)

    # Router decision
    route: str = ""  # 'web', 'rag', 'hybrid', 'closed'

    # Orchestrator plan
    plan: Optional[ReportPlan] = None
    critique_count: int = 0

    # Evidence gathered
    evidence: Annotated[List[EvidenceItem], operator.add] = Field(default_factory=list)

    # Section results (fan-out/fan-in)
    completed_sections: Annotated[List[SectionResult], operator.add] = Field(default_factory=list)

    # Final output
    stitched_report: str = ""
    critique: Optional[CritiqueResult] = None
    final_report: str = ""
    pdf_path: str = ""

    # Streaming status
    status: str = "idle"
    progress_log: Annotated[list[str], operator.add] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
