"""Regression tests for disk caching and fast mode generation."""
import shutil
from pathlib import Path
from unittest.mock import patch

from config.settings import CACHE_DIR
from utils.cache_manager import (
    compute_files_hash,
    has_document_cache,
    save_document_cache,
    load_document_cache,
    get_cache_path,
)
from utils.question_generator import generate_questions


def test_compute_files_hash_deterministic():
    files_a = [("guide.pdf", b"test content 123"), ("notes.txt", b"more notes")]
    files_b = [("guide.pdf", b"test content 123"), ("notes.txt", b"more notes")]
    files_c = [("guide.pdf", b"different content"), ("notes.txt", b"more notes")]

    hash_a = compute_files_hash(files_a)
    hash_b = compute_files_hash(files_b)
    hash_c = compute_files_hash(files_c)

    assert hash_a == hash_b
    assert hash_a != hash_c
    assert len(hash_a) == 32


def test_cache_save_and_load(tmp_path):
    test_hash = "test_hash_12345"
    cache_folder = get_cache_path(test_hash)

    # Clean up before test if exists
    if cache_folder.exists():
        shutil.rmtree(cache_folder)

    try:
        assert not has_document_cache(test_hash)

        saved = save_document_cache(
            doc_hash=test_hash,
            primary_name="guide.pdf",
            primary_text="sample text",
            study_docs=[{"filename": "notes.txt", "text": "extra"}],
            classification={"is_exam_related": True, "confidence": "high"},
            is_exam_doc=True,
            exam_title="GCP PMLE",
            topics=[{"id": 1, "name": "ML Pipelines", "description": "Desc", "subtopics": []}],
            vector_store=None,
        )
        assert saved is True
        assert has_document_cache(test_hash)

        loaded = load_document_cache(test_hash)
        assert loaded is not None
        assert loaded["primary_name"] == "guide.pdf"
        assert loaded["exam_title"] == "GCP PMLE"
        assert len(loaded["topics"]) == 1
        assert loaded["topics"][0]["name"] == "ML Pipelines"
    finally:
        if cache_folder.exists():
            shutil.rmtree(cache_folder)


def test_generate_questions_fast_mode():
    mock_raw_questions = [
        {
            "id": 1,
            "topic": "Feature Engineering",
            "question_type": "single_answer",
            "answer_mode": "single",
            "case_study_context": "",
            "question": "Which BigQuery ML function should you use for bucketization?",
            "choices": {
                "A": "ML.BUCKETIZE",
                "B": "ML.FEATURE_CROSS",
                "C": "ML.POLYNOMIAL_EXPAND",
                "D": "ML.QUANTILE_BUCKETIZE",
            },
            "correct_answer": ["A"],
            "explanation": "ML.BUCKETIZE splits continuous numerical features into buckets.",
            "reference": "https://cloud.google.com/bigquery-ml/docs",
        }
    ]

    with patch("utils.question_generator._invoke_generation", return_value=mock_raw_questions) as mock_invoke:
        results = generate_questions(
            exam_title="GCP PMLE",
            topic={"name": "Feature Engineering", "description": "Transforming data", "subtopics": []},
            num_questions=1,
            fast_mode=True,
        )

        mock_invoke.assert_called_once()
        assert len(results) == 1
        assert results[0]["question"] == "Which BigQuery ML function should you use for bucketization?"
        assert results[0]["correct_answer"] == ["A"]
