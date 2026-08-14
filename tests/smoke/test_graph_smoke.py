"""Smoke tests for the LangGraph question workflow."""

from utils.question_graph import build_question_graph
from utils.question_generator import validate_question


def test_graph_builds_smoke() -> None:
    graph = build_question_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_validate_question_passes_clean_question_smoke() -> None:
    q = {
        "question_type": "single_answer",
        "answer_mode": "single",
        "case_study_context": "",
        "question": "Which managed service serves online predictions?",
        "choices": {"A": "Vertex AI Endpoints", "B": "BigQuery", "C": "Cloud Storage", "D": "Pub/Sub"},
        "correct_answer": ["A"],
        "explanation": "Endpoints serve low-latency online predictions.",
    }
    assert validate_question(q) == []
