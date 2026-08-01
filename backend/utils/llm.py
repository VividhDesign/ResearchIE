"""LLM and embedding model factory."""
from __future__ import annotations
import os
from dotenv import load_dotenv

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
    """Return the embedding model (Gemini text-embedding-004)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
