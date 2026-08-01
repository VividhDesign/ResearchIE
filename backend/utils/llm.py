"""LLM and embedding model factory."""
from __future__ import annotations
import os
from dotenv import load_dotenv

EMBED_VERSION = "v6-huggingface-local"  # No API key needed, free, works on Streamlit Cloud

load_dotenv()


def get_llm(temperature: float = 0.3, structured_output=None):
    """Return the primary LLM (Gemini or Groq) with optional structured output."""
    provider = os.getenv("PRIMARY_LLM", "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )
    else:
        from langchain_groq import ChatGroq
        model = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )

    if structured_output:
        return model.with_structured_output(structured_output)
    return model


def get_fast_llm(temperature: float = 0.1, structured_output=None):
    """Return a fast LLM for quick classification/routing tasks (uses Groq by default)."""
    from langchain_groq import ChatGroq
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )
    if structured_output:
        return model.with_structured_output(structured_output)
    return model


def get_embeddings():
    """Return HuggingFace local embedding model — no API key required.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 (~80MB) which runs locally
    on Streamlit Cloud. Much more reliable than Gemini embeddings which
    require specific API key scopes that many users don't have.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # If user still has old Gemini model name set, override it
    if "embedding" in model_name and ("gemini" in model_name or "text-embedding" in model_name or "gecko" in model_name):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

