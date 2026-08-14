"""Regression tests for question generation JSON parsing and normalization."""

from utils.question_generator import normalize_one, validate_question, _parse_json


def _normalize_questions(items):
    kept = []
    for i, raw in enumerate(items, start=1):
        q = normalize_one(raw, idx=i)
        if q is not None and not validate_question(q):
            kept.append(q)
    return kept


def test_parse_json_strips_markdown_fences_regression() -> None:
    raw = """```json
    {\n  \"status\": \"ok\"\n}
    ```"""
    parsed = _parse_json(raw, fallback={})
    assert parsed == {"status": "ok"}


def test_parse_json_extracts_outer_array_regression() -> None:
    raw = "noise before\n[{\"id\": 1}]\nnoise after"
    parsed = _parse_json(raw, fallback=[])
    assert parsed == [{"id": 1}]


def test_normalize_questions_keeps_valid_single_and_multi_regression() -> None:
    items = [
        {
            "id": 1,
            "topic": "Model deployment",
            "question_type": "single_answer",
            "answer_mode": "single",
            "case_study_context": "",
            "question": "Which service should you use for online inference?",
            "choices": {
                "A": "Vertex AI Endpoint",
                "B": "Cloud Storage",
                "C": "BigQuery",
                "D": "Cloud Run Jobs",
            },
            "correct_answer": ["A"],
            "explanation": "Vertex AI Endpoint is designed for online inference.",
            "reference": "https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions",
        },
        {
            "id": 2,
            "topic": "Monitoring",
            "question_type": "multiple_select",
            "answer_mode": "multi",
            "case_study_context": "",
            "question": "Which TWO actions improve model observability?",
            "choices": {
                "A": "Enable model monitoring",
                "B": "Use random hardcoded labels",
                "C": "Log prediction features",
                "D": "Disable alerts",
            },
            "correct_answer": ["A", "C"],
            "explanation": "Monitoring and feature logging support drift detection.",
            "reference": "https://cloud.google.com/vertex-ai/docs/model-monitoring/overview",
        },
    ]

    normalized = _normalize_questions(items)
    assert len(normalized) == 2
    assert normalized[0]["answer_mode"] == "single"
    assert normalized[0]["correct_answer"] == ["A"]
    assert normalized[1]["answer_mode"] == "multi"
    assert set(normalized[1]["correct_answer"]) == {"A", "C"}


def test_normalize_questions_rejects_invalid_multi_answer_count_regression() -> None:
    items = [
        {
            "id": 1,
            "topic": "Pipelines",
            "question_type": "multiple_select",
            "answer_mode": "multi",
            "question": "Which TWO services should you choose?",
            "choices": {
                "A": "Vertex AI Pipelines",
                "B": "Cloud Functions",
                "C": "Cloud Composer",
                "D": "Dataflow",
            },
            "correct_answer": ["A"],
            "explanation": "",
            "reference": "https://cloud.google.com/vertex-ai/docs/pipelines",
        }
    ]

    normalized = _normalize_questions(items)
    assert normalized == []


def test_normalize_questions_rejects_non_abcd_choices_regression() -> None:
    items = [
        {
            "id": 1,
            "topic": "Training",
            "question_type": "single_answer",
            "answer_mode": "single",
            "question": "What should you do first?",
            "choices": {
                "A": "Analyze drift",
                "B": "Retrain",
                "C": "Rollback",
                "E": "Ignore alerts",
            },
            "correct_answer": ["A"],
            "explanation": "",
            "reference": "https://cloud.google.com/vertex-ai/docs/model-monitoring/overview",
        }
    ]

    normalized = _normalize_questions(items)
    assert normalized == []
