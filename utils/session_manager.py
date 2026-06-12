"""
Centralized Streamlit session state management.

All session_state keys are defined as module-level constants to prevent typos.
Use the get/set helpers instead of accessing st.session_state directly.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

# ── Session State Key Constants ──────────────────────────────────────────────

# Document state
SS_DOCUMENT_TEXT = "document_text"          # str: exam guide raw text
SS_DOCUMENT_NAME = "document_name"          # str: exam guide filename
SS_STUDY_DOCS = "study_docs"                # list[dict]: [{filename, text}] additional docs
SS_VECTOR_STORE = "vector_store"            # FAISS: RAG index (all uploaded docs)
SS_IS_EXAM_DOC = "is_exam_doc"              # bool | None: AI classification result
SS_EXAM_TITLE = "exam_title"               # str: inferred exam title

# Topics state
SS_TOPICS = "topics"                        # list[dict]: extracted topics
SS_SELECTED_TOPICS = "selected_topics"      # list[int]: topic IDs chosen for quiz

# Quiz configuration
SS_NUM_QUESTIONS = "num_questions"          # int: questions per topic
SS_BLOOM_LEVEL = "bloom_level"             # int: Bloom's taxonomy level (3-6)

# Quiz runtime state
SS_QUESTIONS = "questions"                  # list[dict]: generated question objects
SS_CURRENT_Q_IDX = "current_q_idx"         # int: current question index (0-based)
SS_ANSWERS = "answers"                      # dict: {question_id: "A"|"B"|"C"|"D"}
SS_QUIZ_COMPLETE = "quiz_complete"          # bool: True when all questions answered


_DEFAULTS: dict[str, Any] = {
    SS_DOCUMENT_TEXT: "",
    SS_DOCUMENT_NAME: "",
    SS_STUDY_DOCS: [],
    SS_VECTOR_STORE: None,
    SS_IS_EXAM_DOC: None,
    SS_EXAM_TITLE: "",
    SS_TOPICS: [],
    SS_SELECTED_TOPICS: [],
    SS_NUM_QUESTIONS: 5,
    SS_BLOOM_LEVEL: 4,
    SS_QUESTIONS: [],
    SS_CURRENT_Q_IDX: 0,
    SS_ANSWERS: {},
    SS_QUIZ_COMPLETE: False,
}


def init_session() -> None:
    """Initialize all session state keys with defaults (idempotent)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str, default: Any = None) -> Any:
    """Get a session state value."""
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:
    """Set a session state value."""
    st.session_state[key] = value


def reset_quiz() -> None:
    """Clear quiz progress while keeping document, topics, and configuration."""
    st.session_state[SS_QUESTIONS] = []
    st.session_state[SS_CURRENT_Q_IDX] = 0
    st.session_state[SS_ANSWERS] = {}
    st.session_state[SS_QUIZ_COMPLETE] = False


def reset_all() -> None:
    """Full session reset — clears everything and re-initializes defaults."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session()


# ── Convenience predicates ───────────────────────────────────────────────────

def has_document() -> bool:
    return bool(get(SS_DOCUMENT_TEXT))


def has_vector_store() -> bool:
    return get(SS_VECTOR_STORE) is not None


def has_topics() -> bool:
    return bool(get(SS_TOPICS))


def has_questions() -> bool:
    return bool(get(SS_QUESTIONS))


def quiz_in_progress() -> bool:
    return has_questions() and not get(SS_QUIZ_COMPLETE)


def quiz_complete() -> bool:
    return get(SS_QUIZ_COMPLETE, False)
