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
    """Return the embedding model using the google-genai v1 SDK directly.

    Uses google-genai (new SDK, v1 API) instead of langchain's
    GoogleGenerativeAIEmbeddings which internally calls the v1beta endpoint
    and may not find all models.
    """
    from typing import List
    from langchain_core.embeddings import Embeddings
    from google import genai as google_genai

    class _GeminiEmbeddings(Embeddings):
        def __init__(self, api_key: str, model: str):
            self._client = google_genai.Client(api_key=api_key)
            # Strip any 'models/' prefix — the v1 SDK doesn't want it
            self._model = model.replace("models/", "")

        def embed_query(self, text: str) -> List[float]:
            result = self._client.models.embed_content(
                model=self._model,
                contents=text,
            )
            return list(result.embeddings[0].values)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self.embed_query(t) for t in texts]

    api_key = os.getenv("GEMINI_API_KEY", "")
    model   = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    return _GeminiEmbeddings(api_key=api_key, model=model)

