"""RAG ingest pipeline — chunks PDFs/docs and indexes them in FAISS."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = Path(__file__).parent.parent.parent / "vectorstore"


def ingest_documents(file_paths: list[str]) -> int:
    """Chunk and embed documents into the FAISS vector store. Returns number of chunks."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from backend.utils.llm import get_embeddings

    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        separators=["\n\n", "\n", ". ", " "],
    )

    all_docs = []
    for path in file_paths:
        try:
            ext = Path(path).suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            raw_docs = loader.load()
            chunks = splitter.split_documents(raw_docs)
            for chunk in chunks:
                chunk.metadata["source_file"] = Path(path).name
            all_docs.extend(chunks)
        except Exception as e:
            print(f"Failed to ingest {path}: {e}")

    if not all_docs:
        return 0

    # Load existing store or create new
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store_path = str(VECTORSTORE_DIR)

    try:
        store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(all_docs)
    except Exception:
        store = FAISS.from_documents(all_docs, embeddings)

    store.save_local(store_path)
    return len(all_docs)


def clear_vectorstore():
    """Clear the FAISS index."""
    import shutil
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
