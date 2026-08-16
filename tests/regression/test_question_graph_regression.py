"""Regression tests for deterministic validation and the critic-refiner graph routing."""

import utils.question_graph as qg
from utils.question_generator import validate_question


def _valid_question() -> dict:
    return {
        "id": 1,
        "topic": "Serving",
        "question_type": "single_answer",
        "answer_mode": "single",
        "case_study_context": "",
        "question": "Which managed service serves low-latency online predictions?",
        "choices": {"A": "Vertex AI Endpoints", "B": "BigQuery", "C": "Cloud Storage", "D": "Pub/Sub"},
        "correct_answer": ["A"],
        "explanation": "Endpoints serve online predictions with autoscaling.",
        "reference": "https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions",
    }


def test_validate_question_flags_leaked_answer_regression() -> None:
    q = _valid_question()
    q["question"] = "Should you use Vertex AI Endpoints for online predictions?"
    issues = validate_question(q)
    assert any("leaked" in i.lower() for i in issues)


def test_validate_question_flags_duplicate_options_regression() -> None:
    q = _valid_question()
    q["choices"]["B"] = "Vertex AI Endpoints"
    assert any("duplicate" in i.lower() for i in validate_question(q))


def test_validate_question_flags_missing_case_study_context_regression() -> None:
    q = _valid_question()
    q["question_type"] = "case_study"
    q["case_study_context"] = ""
    assert any("case" in i.lower() for i in validate_question(q))


def test_judge_node_accepts_high_score_regression(monkeypatch) -> None:
    monkeypatch.setattr(
        qg,
        "_run_json_chain",
        lambda *a, **k: {"overall_score": 9.2, "decision": "ACCEPT", "critical_issues": [], "improvement_instructions": []},
    )
    state = {"question": _valid_question(), "deterministic_issues": [], "critic": {}, "iteration": 0}
    out = qg.judge_node(state)
    assert out["final_status"] == "ACCEPT"


def test_judge_node_rejects_at_max_iteration_regression(monkeypatch) -> None:
    monkeypatch.setattr(
        qg,
        "_run_json_chain",
        lambda *a, **k: {"overall_score": 4.0, "decision": "IMPROVE", "critical_issues": ["ambiguous"], "improvement_instructions": ["fix"]},
    )
    state = {
        "question": _valid_question(),
        "deterministic_issues": ["Choices contain duplicate option text."],
        "critic": {},
        "iteration": qg.QUESTION_MAX_ITERATIONS,
    }
    out = qg.judge_node(state)
    assert out["final_status"] == "REJECT"


def test_judge_node_routes_to_improve_when_not_terminal_regression(monkeypatch) -> None:
    monkeypatch.setattr(
        qg,
        "_run_json_chain",
        lambda *a, **k: {"overall_score": 5.0, "decision": "IMPROVE", "critical_issues": ["too easy"], "improvement_instructions": ["add constraint"]},
    )
    state = {"question": _valid_question(), "deterministic_issues": [], "critic": {}, "iteration": 0}
    out = qg.judge_node(state)
    assert "final_status" not in out
    assert qg._route_after_judge(out) == "improve"


def _fake_json_chain(system, user, variables, temperature=0, fast=False):
    s = system.lower()
    if "quality judge" in s:
        return {"overall_score": 9.5, "decision": "ACCEPT", "critical_issues": [], "improvement_instructions": []}
    if "improver" in s:
        return _valid_question()
    return {"scores": {"technical": 9}, "issues": []}


def test_pipeline_accepts_valid_question_regression(monkeypatch) -> None:
    monkeypatch.setattr(qg, "_invoke_generation", lambda ctx, num_questions, temperature=0.3: [_valid_question()])
    monkeypatch.setattr(qg, "gather_verification_evidence", lambda ctx, question: "evidence")
    monkeypatch.setattr(qg, "_run_json_chain", _fake_json_chain)
    qg._COMPILED_GRAPH = None

    result = qg.run_question_pipeline({"exam_title": "PMLE", "topic": {"name": "Serving"}})
    assert result is not None
    assert result["correct_answer"] == ["A"]


def test_pipeline_rejects_unfixable_question_regression(monkeypatch) -> None:
    bad = _valid_question()
    bad["correct_answer"] = ["A", "B"]  # single-answer with two correct -> always invalid

    def low_judge(system, user, variables, temperature=0, fast=False):
        s = system.lower()
        if "quality judge" in s:
            return {"overall_score": 3.0, "decision": "IMPROVE", "critical_issues": ["broken"], "improvement_instructions": ["fix"]}
        if "improver" in s:
            return bad  # improver keeps returning the broken question
        return {"scores": {"technical": 3}, "issues": ["wrong"]}

    monkeypatch.setattr(qg, "_invoke_generation", lambda ctx, num_questions, temperature=0.3: [bad])
    monkeypatch.setattr(qg, "gather_verification_evidence", lambda ctx, question: "evidence")
    monkeypatch.setattr(qg, "_run_json_chain", low_judge)
    qg._COMPILED_GRAPH = None

    result = qg.run_question_pipeline({"exam_title": "PMLE", "topic": {"name": "Serving"}})
    assert result is None


def test_evidence_gathering_uses_search_results_regression(monkeypatch) -> None:
    monkeypatch.setattr(
        qg,
        "search_topic",
        lambda exam_title, fragment: [
            {"title": "Online predictions", "url": "https://cloud.google.com/x", "body": "Vertex AI Endpoints serve online predictions."}
        ],
    )
    evidence = qg.gather_verification_evidence(
        {"exam_title": "PMLE", "topic": {"name": "Serving"}}, _valid_question()
    )
    assert "cloud.google.com" in evidence
    assert "Vertex AI Endpoints" in evidence


def test_validation_agents_require_search_evidence_regression() -> None:
    for prompt in (qg._CRITIC_USER, qg._JUDGE_USER, qg._IMPROVER_USER):
        assert "{evidence}" in prompt
    for system in (qg._CRITIC_SYSTEM, qg._JUDGE_SYSTEM, qg._IMPROVER_SYSTEM):
        assert "prior knowledge" in system.lower()
