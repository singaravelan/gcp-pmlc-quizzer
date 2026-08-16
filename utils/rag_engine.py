"""
RAG (Retrieval-Augmented Generation) engine using FAISS vector store.

Handles document chunking, embedding, and similarity retrieval.
The FAISS index lives in Streamlit session_state for the duration of a session.
"""
from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.ai_client import get_embeddings
from config.settings import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _text_to_documents(text: str, source: str) -> list[Document]:
    """Split a single document's text into LangChain Document chunks."""
    splitter = _make_splitter()
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"source": source, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]


def build_vector_store(
    texts: list[str],
    filenames: list[str],
) -> FAISS:
    """
    Build a FAISS vector store from a list of document texts.

    Args:
        texts: Plain-text content of each document.
        filenames: Corresponding filenames (used as metadata source labels).

    Returns:
        A populated FAISS vector store ready for similarity search.
    """
    all_docs: list[Document] = []
    for text, filename in zip(texts, filenames):
        all_docs.extend(_text_to_documents(text, source=filename))

    if not all_docs:
        raise ValueError("No document chunks were created. Check that your files contain text.")

    embeddings = get_embeddings()
    store = FAISS.from_documents(all_docs, embeddings)
    return store


def retrieve_context(
    query: str,
    vector_store: FAISS,
    k: int = RAG_TOP_K,
) -> str:
    """
    Perform similarity search and return top-k chunks as a single text block.

    Args:
        query: The search query (typically the topic name + exam title).
        vector_store: The FAISS store to search.
        k: Number of top chunks to retrieve.

    Returns:
        Combined text of top-k retrieved chunks, labelled by source.
    """
    results = vector_store.similarity_search(query, k=k)
    if not results:
        return ""

    parts: list[str] = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def count_chunks(vector_store: FAISS) -> int:
    """Return the number of chunks stored in the FAISS index."""
    try:
        return vector_store.index.ntotal
    except Exception:
        return 0
