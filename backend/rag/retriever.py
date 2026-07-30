"""RAG retriever node — queries FAISS for relevant document chunks."""
from __future__ import annotations
import os
from pathlib import Path
from backend.state import ResearchState, EvidenceItem

VECTORSTORE_DIR = Path(__file__).parent.parent.parent / "vectorstore"


def rag_research_node(state: ResearchState) -> dict:
    """Retrieve relevant chunks from FAISS and add to evidence pack."""
    from langchain_community.vectorstores import FAISS
    from backend.utils.llm import get_embeddings

    if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
        return {
            "progress_log": ["RAG: No vectorstore found, skipping document retrieval"],
        }

    embeddings = get_embeddings()
    top_k = int(os.getenv("RAG_TOP_K", 5))
    evidence = []
    progress = []

    try:
        store = FAISS.load_local(
            str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
        )

        queries = []
        if state.plan and state.plan.rag_queries:
            queries = state.plan.rag_queries[:5]
        else:
            queries = [state.topic]

        seen_contents = set()
        for query in queries:
            docs = store.similarity_search(query, k=top_k)
            for doc in docs:
                content_key = doc.page_content[:100]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    evidence.append(EvidenceItem(
                        source=doc.metadata.get("source_file", "uploaded_document"),
                        title=f"Document: {doc.metadata.get('source_file', 'Unknown')}",
                        content=doc.page_content,
                        source_type="rag",
                    ))
            progress.append(f"RAG: '{query}' → {len(docs)} chunks retrieved")

    except Exception as e:
        progress.append(f"RAG retrieval error: {e}")

    return {
        "evidence": evidence,
        "status": f"📂 RAG retrieval complete: {len(evidence)} document chunks",
        "progress_log": progress,
    }
