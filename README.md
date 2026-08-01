# 🔬 Research Intelligence Engine (RIE)

> An autonomous multi-agent research report generator built with LangGraph, featuring Orchestrator-Worker-Critic architecture, RAG over private documents, Tavily web search, and a real-time Streamlit UI.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://researchie.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)

---

## Architecture

```
User Topic → Router → [Pre-Research] → Orchestrator (Planner)
                                              ↓
                                    Fan-out Section Writers (parallel)
                                              ↓
                                    Stitcher → Critic → [Replan loop]
                                              ↓
                                    Diagram Node → Finalize → PDF
```

## Features

- **Orchestrator-Worker-Critic** LangGraph multi-agent pipeline
- **Parallel section writing** via LangGraph `Send()` fan-out
- **RAG over uploaded PDFs** using FAISS + Gemini embeddings
- **Web search** via Tavily API
- **Quality critique loop** — auto-retries if quality score < 7/10
- **Mermaid diagram generation** for visual sections
- **PDF export** via ReportLab
- **Real-time streaming** progress in Streamlit UI

---

## Local Setup

### 1. Clone & install dependencies
```bash
git clone https://github.com/VividhDesign/ResearchIE.git
cd ResearchIE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your actual API keys:
# GEMINI_API_KEY=...
# GROQ_API_KEY=...
# TAVILY_API_KEY=...
```

### 3. Run the app
```bash
streamlit run frontend/app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. **Fork / push** this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set:
   - **Repository**: `VividhDesign/ResearchIE`
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GROQ_API_KEY = "your_groq_api_key"
TAVILY_API_KEY = "your_tavily_api_key"

# Optional — defaults shown
PRIMARY_LLM = "gemini"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "models/text-embedding-004"
MAX_CRITIQUE_RETRIES = "2"
MAX_SECTIONS = "8"
```

5. Click **Deploy** 🚀

> **Note**: The FAISS vector store is ephemeral on Streamlit Cloud (in-memory per session). Uploaded documents are indexed per session only.

---

## API Keys

| Service | Purpose | Get it |
|---------|---------|--------|
| Gemini | LLM + Embeddings | [aistudio.google.com](https://aistudio.google.com) |
| Groq | Fast LLM (optional) | [console.groq.com](https://console.groq.com) |
| Tavily | Web search | [tavily.com](https://tavily.com) |
