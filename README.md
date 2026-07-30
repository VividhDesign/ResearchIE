# Research Intelligence Engine (RIE)

> An autonomous multi-agent research report generator built with LangGraph, featuring Orchestrator-Worker-Critic architecture, RAG over private documents, Tavily web search, and a real-time Streamlit UI.

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

## Setup

### 1. Install dependencies
```bash
cd /Users/vividhyadav/Projects/ResearchIE
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

## Features
- **Orchestrator-Worker-Critic** LangGraph multi-agent pipeline
- **Parallel section writing** via LangGraph `Send()` fan-out
- **RAG over uploaded PDFs** using FAISS + Gemini embeddings
- **Web search** via Tavily API
- **Quality critique loop** — auto-retries if quality score < 7/10
- **Mermaid diagram generation** for visual sections
- **PDF export** via ReportLab
- **Real-time streaming** progress in Streamlit UI
