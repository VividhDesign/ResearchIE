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
    """Return the embedding model using a direct REST API call.
    
    This completely bypasses the Python SDKs to avoid the 'v1beta' 404 bugs 
    that happen when the SDKs hardcode old API versions.
    """
    from typing import List
    from langchain_core.embeddings import Embeddings
    import requests

    class _GeminiRESTEmbeddings(Embeddings):
        def __init__(self, api_key: str, model: str):
            self.api_key = api_key
            self.model = model.replace("models/", "")
            # Force v1 API
            self.url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:embedContent?key={self.api_key}"

        def embed_query(self, text: str) -> List[float]:
            resp = requests.post(
                self.url,
                json={
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]}
                }
            )
            if resp.status_code != 200:
                raise Exception(f"Embedding failed: {resp.text}")
            return resp.json()["embedding"]["values"]

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # For simplicity, embed sequentially (or you could use batchEmbedContents)
            return [self.embed_query(t) for t in texts]

    api_key = os.getenv("GEMINI_API_KEY", "")
    model   = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    return _GeminiRESTEmbeddings(api_key=api_key, model=model)

