"""
Research Intelligence Engine — Streamlit UI
A beautiful, real-time streaming research report generator.
"""
import streamlit as st
import sys
import os
import time
import tempfile
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from frontend.history import save_report, list_past_reports, load_report

# ─── Streamlit Cloud: sync st.secrets → os.environ ─────────────────────────────
# On Streamlit Cloud there is no .env file; API keys live in st.secrets.
# We copy them into os.environ so all backend os.getenv() calls work unchanged.
_SECRET_KEYS = [
    "GEMINI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY",
    "PRIMARY_LLM", "GEMINI_MODEL", "GROQ_MODEL", "EMBEDDING_MODEL",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "RAG_TOP_K",
    "MAX_CRITIQUE_RETRIES", "MAX_SECTIONS",
]
for _k in _SECRET_KEYS:
    if _k not in os.environ:
        try:
            os.environ[_k] = str(st.secrets[_k])
        except (KeyError, FileNotFoundError):
            pass  # Not present in secrets — will fail gracefully at runtime

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Intelligence Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #000000; color: #ffffff; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #333333;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #ffffff; }

    /* Main header */
    .rie-header {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .rie-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        border-radius: 50%;
    }
    .rie-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        padding: 0;
    }
    .rie-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
        font-weight: 400;
    }

    /* Progress log */
    .progress-box {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #cccccc;
        max-height: 280px;
        overflow-y: auto;
        line-height: 1.8;
    }
    .progress-item { color: #ffffff; }
    .progress-item.error { color: #ff4444; }

    /* Section card */
    .section-card {
        background: #000000;
        border: 1px solid #333333;
        border-left: 3px solid #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        transition: all 0.3s ease;
    }
    .section-card:hover { border-left-color: #888888; }
    .section-title { color: #ffffff; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }

    /* Report display */
    .report-container {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 32px 40px;
        line-height: 1.8;
        color: #e6edf3;
    }
    .report-container h1 { color: #ffffff; border-bottom: 1px solid #333333; padding-bottom: 12px; }
    .report-container h2 { color: #eeeeee; }
    .report-container h3 { color: #dddddd; }
    .report-container code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9em; }
    .report-container blockquote { border-left: 3px solid #888888; padding-left: 16px; color: #aaaaaa; margin: 16px 0; }

    /* Metric cards */
    .metric-row { display: flex; gap: 16px; margin: 20px 0; }
    .metric-card {
        flex: 1;
        background: #000000;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #ffffff; }
    .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

    /* Buttons */
    .stButton > button {
        background: #222222;
        color: white;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: #333333;
        border-color: #666666;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #000000; border-bottom: 1px solid #333333; }
    .stTabs [data-baseweb="tab"] { color: #888888; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #ffffff; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #000000;
        border: 2px dashed #333333;
        border-radius: 10px;
        padding: 16px;
    }

    /* Input fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background: #000000 !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }
    .stTextInput input:focus { border-color: #ffffff !important; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Diagram container */
    .diagram-box {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 20px;
        margin: 12px 0;
        overflow-x: auto;
    }

    /* Fix dropdown scroll */
    ul[data-baseweb="menu"] {
        max-height: 250px !important;
        overflow-y: auto !important;
    }
    div[data-baseweb="popover"] {
        max-height: 250px !important;
        overflow-y: auto !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper functions ──────────────────────────────────────────────────────────

def render_mermaid(mermaid_code: str):
    """Render a Mermaid diagram in Streamlit."""
    html = f"""
    <div class="diagram-box">
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
        <div class="mermaid">
        {mermaid_code}
        </div>
    </div>
    """
    st.components.v1.html(html, height=350)


def display_progress(logs: list[str], placeholder):
    """Display progress logs in the terminal-style box."""
    log_html = "<br>".join([
        f'<span class="progress-item {"error" if "fail" in l.lower() or "error" in l.lower() else ""}">{l}</span>'
        for l in logs
    ])
    placeholder.markdown(
        f'<div class="progress-box">{log_html}</div>',
        unsafe_allow_html=True
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_gemini_models(api_key: str) -> list[str]:
    """Fetch available text generation models from Gemini API."""
    fallback = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    if not api_key: return fallback
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            models = res.json().get("models", [])
            valid = [m["name"].replace("models/", "") for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
            return sorted(valid, reverse=True) if valid else fallback
    except:
        pass
    return fallback

@st.cache_data(ttl=3600, show_spinner=False)
def get_groq_models(api_key: str) -> list[str]:
    """Fetch available models from Groq API."""
    fallback = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    if not api_key: return fallback
    try:
        res = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        if res.status_code == 200:
            models = res.json().get("data", [])
            valid = [m["id"] for m in models]
            return sorted(valid) if valid else fallback
    except:
        pass
    return fallback


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # API Key status
    st.markdown("### 🔑 API Keys")
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))

    st.markdown(f"{'✅' if gemini_ok else '❌'} Gemini {'Connected' if gemini_ok else 'Not configured'}")
    st.markdown(f"{'✅' if groq_ok else '❌'} Groq {'Connected' if groq_ok else 'Not configured'}")
    st.markdown(f"{'✅' if tavily_ok else '❌'} Tavily {'Connected' if tavily_ok else 'Not configured'}")

    st.markdown("---")
    st.markdown("### 📂 Upload Documents (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload documents to use as a private knowledge base for your report",
        key="doc_upload",
    )

    if uploaded_files:
        if st.button("Index Documents", key="index_docs"):
            from backend.rag.ingest import ingest_documents
            with st.spinner("Indexing documents..."):
                paths = []
                for f in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(f.name).suffix
                    )
                    tmp.write(f.read())
                    tmp.close()
                    paths.append(tmp.name)
                try:
                    n_chunks = ingest_documents(paths)
                    if n_chunks == 0:
                        st.warning("No chunks were indexed. Check your documents.")
                    else:
                        st.success(f"Indexed {n_chunks} chunks from {len(paths)} file(s)")
                        st.session_state["doc_paths"] = paths
                except Exception as e:
                    from backend.utils.llm import EMBED_VERSION
                    st.error(f"Indexing failed [{EMBED_VERSION}]: {type(e).__name__}: {str(e)}")
                    st.info("Check that your GEMINI_API_KEY is valid and has access to the Embeddings API.")

    st.markdown("---")
    st.markdown("### 📚 Past Research")
    from frontend.history import list_past_reports, load_report, delete_report, clear_all_reports
    past_reports = list_past_reports()
    
    if past_reports:
        options = ["Current / New Session"] + [f"{r['topic'][:35]}..." for r in past_reports]
        selected_hist = st.radio("History", options, label_visibility="collapsed")
        
        if selected_hist != "Current / New Session":
            idx = options.index(selected_hist) - 1
            rep = past_reports[idx]
            
            # Show delete button for selected report
            if st.button("🗑️ Delete Selected", key="del_selected", use_container_width=True):
                if delete_report(rep['filename']):
                    if st.session_state.get("current_loaded_history") == rep['id']:
                        st.session_state["current_loaded_history"] = "current"
                        if "report_result" in st.session_state:
                            del st.session_state["report_result"]
                        st.session_state["show_report"] = False
                    st.rerun()
                    
            if st.session_state.get("current_loaded_history") != rep['id']:
                data = load_report(rep['filename'])
                if data:
                    st.session_state["report_result"] = data["final_state"]
                    st.session_state["show_report"] = True
                    st.session_state["all_logs"] = data.get("logs", [])
                    st.session_state["current_loaded_history"] = rep['id']
                    st.rerun()
        else:
            if st.session_state.get("current_loaded_history") not in [None, "current"]:
                # User clicked back to new session
                st.session_state["current_loaded_history"] = "current"
                if "report_result" in st.session_state:
                    del st.session_state["report_result"]
                st.session_state["show_report"] = False
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History", key="clear_history", use_container_width=True):
            clear_all_reports()
            st.session_state["current_loaded_history"] = "current"
            if "report_result" in st.session_state:
                del st.session_state["report_result"]
            st.session_state["show_report"] = False
            st.rerun()
    else:
        st.info("No past research found.")

    st.markdown("---")
    st.markdown("### 🤖 Model Settings")

    _GEMINI_MODELS = get_gemini_models(os.getenv("GEMINI_API_KEY", ""))
    _GROQ_MODELS = get_groq_models(os.getenv("GROQ_API_KEY", ""))

    primary_llm = st.selectbox(
        "Provider",
        ["gemini", "groq"],
        index=0,
        key="primary_llm",
        help="Choose the LLM provider for report generation",
    )
    os.environ["PRIMARY_LLM"] = primary_llm

    if primary_llm == "gemini":
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if default_model not in _GEMINI_MODELS:
            _GEMINI_MODELS.insert(0, default_model)
        
        chosen_model = st.selectbox(
            "Gemini Model",
            _GEMINI_MODELS,
            index=_GEMINI_MODELS.index(default_model),
            key="gemini_model_select",
            help="Select from available Gemini models",
        )
        os.environ["GEMINI_MODEL"] = chosen_model
        st.caption(f"🟣 Using **{chosen_model}**")
    else:
        default_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if default_model not in _GROQ_MODELS:
            _GROQ_MODELS.insert(0, default_model)
            
        chosen_model = st.selectbox(
            "Groq Model",
            _GROQ_MODELS,
            index=_GROQ_MODELS.index(default_model),
            key="groq_model_select",
            help="Select from available Groq models",
        )
        os.environ["GROQ_MODEL"] = chosen_model
        st.caption(f"🟠 Using **{chosen_model}**")

    st.markdown("---")
    st.markdown("### 🗑️ Manage Knowledge Base")
    if st.button("Clear Vector Store", key="clear_vs"):
        from backend.rag.ingest import clear_vectorstore
        clear_vectorstore()
        st.success("Vector store cleared!")

    st.markdown("---")
    st.markdown("""
    <div style="color: #475569; font-size: 0.75rem; text-align: center;">
    Research Intelligence Engine<br>
    Powered by LangGraph + Gemini + Groq
    </div>
    """, unsafe_allow_html=True)


# ─── Main layout ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="rie-header">
    <h1 class="rie-title">🔬 Research Intelligence Engine</h1>
    <p class="rie-subtitle">
        Autonomous multi-agent research reports powered by LangGraph · Orchestrator-Worker-Critic architecture · RAG + Web Search
    </p>
</div>
""", unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_area(
        "📌 Research Topic",
        placeholder="e.g., 'LangGraph vs AutoGen: Technical Deep-Dive for AI Engineers'\n'State of Agentic AI in 2025: Key Frameworks and Patterns'\n'Vector Databases Compared: Pinecone vs Weaviate vs FAISS'",
        height=100,
        key="topic_input",
    )

with col2:
    audience = st.selectbox(
        "👥 Target Audience",
        ["ML/AI Engineers", "Software Architects", "Startup Founders", "Product Managers", "General Technical"],
        key="audience_select",
    )
    tone = st.selectbox(
        "🎨 Report Tone",
        ["Analytical", "Technical Deep-Dive", "Executive Summary", "Academic"],
        key="tone_select",
    )

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    generate_btn = st.button("🚀 Generate Research Report", key="generate_btn", use_container_width=True)
with col_btn2:
    if st.button("📋 Example Topics", key="examples_btn", use_container_width=True):
        st.session_state["show_examples"] = not st.session_state.get("show_examples", False)
with col_btn3:
    if st.button("🔄 Reset", key="reset_btn", use_container_width=True):
        for key in ["report_result", "show_report"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Example topics
if st.session_state.get("show_examples"):
    st.markdown("""
    <div style="background: #111128; border: 1px solid #1e1e3f; border-radius: 10px; padding: 16px; margin: 12px 0;">
    <b style="color: #a78bfa;">💡 Example Topics to try:</b><br><br>
    • <code>LangGraph vs AutoGen vs CrewAI: A Technical Comparison for Production AI Systems</code><br>
    • <code>Vector Databases in 2025: Pinecone vs Weaviate vs FAISS vs Chroma</code><br>
    • <code>The Rise of Agentic AI: Patterns, Pitfalls, and Production Lessons</code><br>
    • <code>RAG vs Fine-Tuning: When to Use Each Approach for LLM Applications</code><br>
    • <code>Kubernetes vs Serverless for ML Model Serving: A Cost and Performance Analysis</code>
    </div>
    """, unsafe_allow_html=True)


# ─── Generation flow ──────────────────────────────────────────────────────────

if generate_btn and topic.strip():
    from backend.graph import get_graph
    from backend.state import ResearchState

    graph = get_graph()

    # Status area
    st.markdown("---")
    st.markdown("### ⚡ Agent Progress")

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    section_placeholder = st.empty()

    all_logs = []
    sections_seen = []
    final_state = None

    initial_state = ResearchState(
        topic=topic.strip(),
        audience=audience,
        tone=tone.lower().replace(" ", "_"),
        uploaded_doc_paths=st.session_state.get("doc_paths", []),
    )

    try:
        cumulative_state = initial_state.model_dump()
        
        def _to_dict(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, list):
                return [_to_dict(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj

        # Stream graph execution
        for event in graph.stream(
            cumulative_state,
            stream_mode="updates",
            config={"recursion_limit": 50},
        ):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    # Aggregate state manually since stream_mode="updates" only yields diffs
                    for key, value in node_output.items():
                        clean_value = _to_dict(value)
                        if key in ["evidence", "completed_sections", "progress_log"]:
                            if key not in cumulative_state or not cumulative_state[key]:
                                cumulative_state[key] = []
                            cumulative_state[key].extend(clean_value)
                        else:
                            cumulative_state[key] = clean_value

                    # Update status
                    status = node_output.get("status", "")
                    if status:
                        status_placeholder.markdown(
                            f'<div class="status-badge">⚡ {status}</div>',
                            unsafe_allow_html=True
                        )

                    # Collect logs
                    logs = node_output.get("progress_log", [])
                    if logs:
                        all_logs.extend([f"[{node_name}] {l}" for l in logs])
                        display_progress(all_logs, progress_placeholder)

                    # Show sections as they appear
                    new_sections = node_output.get("completed_sections", [])
                    if new_sections:
                        sections_seen.extend(new_sections)
                        section_html = ""
                        for s in sections_seen[-3:]:  # Show last 3
                            sec_obj = s if isinstance(s, dict) else s.model_dump() if hasattr(s, 'model_dump') else {}
                            title = sec_obj.get('title', str(s)[:50]) if sec_obj else str(s)[:50]
                            wc = sec_obj.get('word_count', '?') if sec_obj else '?'
                            section_html += f"""
                            <div class="section-card">
                                <div class="section-title">✍️ {title}</div>
                                <div style="color: #64748b; font-size: 0.8rem;">~{wc} words written</div>
                            </div>
                            """
                        section_placeholder.markdown(section_html, unsafe_allow_html=True)

                    # Capture final state pieces
                    if "final_report" in node_output and node_output["final_report"]:
                        final_state = cumulative_state

        # Store result
        if final_state:
            st.session_state["report_result"] = final_state
            st.session_state["show_report"] = True
            st.session_state["all_logs"] = all_logs
            st.session_state["current_loaded_history"] = "current"
            
            # Save to persistent history
            save_report(topic.strip(), final_state, all_logs)
            
            st.rerun()

    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            st.error("❌ **API Rate Limit Reached!**")
            st.warning(
                "You have exhausted your free tier quota for the selected model (likely Gemini). "
                "**Please scroll down to '🤖 Model Settings' and switch the Provider to 'groq'** to continue generating reports using Llama models!"
            )
        else:
            st.error(f"❌ Generation failed: {e}")
            import traceback
            with st.expander("Show detailed error log"):
                st.code(traceback.format_exc(), language="python")

elif generate_btn:
    st.warning("⚠️ Please enter a research topic!")


# ─── Report display ───────────────────────────────────────────────────────────

if st.session_state.get("show_report") and "report_result" in st.session_state:
    result = st.session_state["report_result"]
    report_md = result.get("final_report", "")
    pdf_path = result.get("pdf_path", "")

    st.markdown("---")

    # Metrics row
    word_count = len(report_md.split())
    section_count = report_md.count("## ")
    all_logs = st.session_state.get("all_logs", [])

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{word_count:,}</div>
            <div class="metric-label">Total Words</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{section_count}</div>
            <div class="metric-label">Sections Generated</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(all_logs)}</div>
            <div class="metric-label">Agent Steps</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{"✅" if pdf_path else "⚠️"}</div>
            <div class="metric-label">PDF {'Ready' if pdf_path else 'Unavailable'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for Plan, Evidence, Markdown Preview, Diagrams, Logs
    tab_plan, tab_evidence, tab_preview, tab_diagrams, tab_logs = st.tabs([
        "🧩 Plan", "📚 Evidence", "📝 Markdown Preview", "🖼️ Diagrams", "📜 Logs"
    ])

    with tab_plan:
        st.markdown("### 🧩 Research Plan")
        plan = result.get("plan")
        if plan:
            st.json(plan)
        else:
            st.info("No plan data available.")

    with tab_evidence:
        st.markdown("### 📚 Gathered Evidence")
        evidence = result.get("evidence", [])
        if evidence:
            for i, ev in enumerate(evidence):
                with st.expander(f"Evidence {i+1}: {ev.get('title', 'Unknown')}"):
                    st.markdown(f"**Source**: {ev.get('source', 'Unknown')}")
                    st.markdown(f"**Type**: {ev.get('source_type', 'Unknown')}")
                    st.markdown(f"**Content snippet**:\n> {ev.get('content', '')}")
        else:
            st.info("No evidence gathered.")

    with tab_preview:
        # Put download buttons at the top of preview
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📄 Download Markdown",
                data=report_md,
                file_name="research_report.md",
                mime="text/markdown",
                key="dl_md",
                use_container_width=True,
            )
        with col_dl2:
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📕 Download PDF",
                        data=f.read(),
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                        key="dl_pdf",
                        use_container_width=True,
                    )

        st.markdown(
            f'<div class="report-container">{__import__("markdown").markdown(report_md, extensions=["tables", "fenced_code"])}</div>',
            unsafe_allow_html=True,
        )

    with tab_diagrams:
        st.markdown("### 🖼️ Generated Diagrams")
        
        # Render any Mermaid diagrams from the report
        import re
        mermaid_blocks = re.findall(r'```mermaid\n(.*?)```', report_md, re.DOTALL)
        if mermaid_blocks:
            st.markdown(f"**{len(mermaid_blocks)} Mermaid diagram(s):**")
            for i, block in enumerate(mermaid_blocks):
                st.markdown(f"**Diagram {i+1}:**")
                render_mermaid(block)
        else:
            st.info("No diagrams were generated for this report.")

    with tab_logs:
        st.markdown("### 📜 Full Agent Execution Log")
        log_text = "\n".join(all_logs)
        st.code(log_text, language="text")

