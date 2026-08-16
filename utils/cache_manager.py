"""
Cache manager for document processing results (topics, metadata, and FAISS vector store).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config.settings import CACHE_DIR
from utils.rag_engine import save_vector_store, load_vector_store


def compute_files_hash(files_data: list[tuple[str, bytes]]) -> str:
    """
    Compute a deterministic MD5 hash across all uploaded filenames and file bytes.

    Args:
        files_data: List of (filename, file_bytes) tuples.

    Returns:
        Hex digest string representing the unique combination of uploaded documents.
    """
    hasher = hashlib.md5()
    for name, content in files_data:
        hasher.update(name.encode("utf-8", errors="ignore"))
        hasher.update(content)
    return hasher.hexdigest()


def get_cache_path(doc_hash: str) -> Path:
    """Return the cache directory path for a given document hash."""
    return CACHE_DIR / doc_hash


def has_document_cache(doc_hash: str) -> bool:
    """Check whether valid cached metadata exists for this document hash."""
    cache_dir = get_cache_path(doc_hash)
    meta_file = cache_dir / "metadata.json"
    return meta_file.exists()


def load_document_cache(doc_hash: str) -> dict[str, Any] | None:
    """
    Load cached metadata and FAISS vector store for a document hash.

    Returns:
        Dict with keys: primary_name, primary_text, study_docs, classification,
        is_exam_doc, exam_title, topics, vector_store (FAISS instance or None).
    """
    cache_dir = get_cache_path(doc_hash)
    meta_file = cache_dir / "metadata.json"
    if not meta_file.exists():
        return None

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        faiss_dir = cache_dir / "faiss_index"
        vector_store = None
        if faiss_dir.exists():
            vector_store = load_vector_store(faiss_dir)

        data["vector_store"] = vector_store
        return data
    except Exception:
        return None


def save_document_cache(
    doc_hash: str,
    primary_name: str,
    primary_text: str,
    study_docs: list[dict[str, str]],
    classification: dict[str, Any],
    is_exam_doc: bool,
    exam_title: str,
    topics: list[dict[str, Any]],
    vector_store: Any | None = None,
) -> bool:
    """
    Save parsed document metadata, topics, and FAISS index to the cache directory.
    """
    try:
        cache_dir = get_cache_path(doc_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "primary_name": primary_name,
            "primary_text": primary_text,
            "study_docs": study_docs,
            "classification": classification,
            "is_exam_doc": is_exam_doc,
            "exam_title": exam_title,
            "topics": topics,
        }

        meta_file = cache_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if vector_store is not None:
            faiss_dir = cache_dir / "faiss_index"
            save_vector_store(vector_store, faiss_dir)

        return True
    except Exception:
        return False


def list_cached_documents() -> list[dict[str, Any]]:
    """
    List all cached document sets found in CACHE_DIR, sorted from newest to oldest.
    """
    if not CACHE_DIR.exists():
        return []

    caches = []
    for sub in CACHE_DIR.iterdir():
        if sub.is_dir():
            meta_file = sub / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    caches.append({
                        "hash": sub.name,
                        "primary_name": meta.get("primary_name", "Unknown File"),
                        "exam_title": meta.get("exam_title", "Unknown Exam"),
                        "topics_count": len(meta.get("topics", [])),
                        "mtime": meta_file.stat().st_mtime,
                    })
                except Exception:
                    pass

    caches.sort(key=lambda x: x["mtime"], reverse=True)
    return caches


def get_latest_cached_document() -> dict[str, Any] | None:
    """
    Load the most recently cached document package if available.
    """
    cached_list = list_cached_documents()
    if not cached_list:
        return None
    latest_hash = cached_list[0]["hash"]
    return load_document_cache(latest_hash)


def load_cache_into_session(cached: dict[str, Any]) -> bool:
    """
    Populate Streamlit session state from a cached document dictionary.
    """
    if not cached or not cached.get("topics"):
        return False

    from utils.session_manager import (
        set_state,
        SS_DOCUMENT_TEXT,
        SS_DOCUMENT_NAME,
        SS_STUDY_DOCS,
        SS_IS_EXAM_DOC,
        SS_EXAM_TITLE,
        SS_TOPICS,
        SS_VECTOR_STORE,
    )

    set_state(SS_DOCUMENT_TEXT, cached.get("primary_text", ""))
    set_state(SS_DOCUMENT_NAME, cached.get("primary_name", ""))
    set_state(SS_STUDY_DOCS, cached.get("study_docs", []))
    set_state(SS_IS_EXAM_DOC, cached.get("is_exam_doc", True))
    set_state(SS_EXAM_TITLE, cached.get("exam_title", "Exam"))
    set_state(SS_TOPICS, cached.get("topics", []))
    set_state(SS_VECTOR_STORE, cached.get("vector_store"))
    return True

