"""
LangGraph critic-refiner workflow for question generation.

Flow: generate -> deterministic gate -> critic -> judge -> (improve -> loop) | finalize.
Keeps the same output schema as utils.question_generator so callers are unaffected.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from config.settings import QUESTION_MAX_ITERATIONS, QUESTION_ACCEPT_SCORE
from utils.ai_client import get_llm
from utils.web_search import search_topic
from utils.question_generator import (
    _invoke_generation,
    _parse_json,
    normalize_one,
    validate_question,
)


class QuestionState(TypedDict, total=False):
    ctx: dict[str, Any]
    question: dict[str, Any]
    deterministic_issues: list[str]
    evidence: str
    critic: dict[str, Any]
    judge: dict[str, Any]
    iteration: int
    final_status: str  # "ACCEPT" | "REJECT"


# ── Prompts ──────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a strict exam-item critic for GCP Professional ML Engineer practice questions.
Evaluate one question across several dimensions and report concrete issues.
Base your judgment STRICTLY on the SEARCH EVIDENCE provided. Do not rely on your own
prior knowledge. If the evidence does not support the marked correct answer, lower the
technical score and record an issue.
Return ONLY valid JSON. No prose, no markdown fences."""

_CRITIC_USER = """\
Critique this question and its choices using ONLY the search evidence.

SEARCH EVIDENCE (from DuckDuckGo, official GCP docs):
---
{evidence}
---

QUESTION JSON:
{question_json}

Return ONLY this JSON:
{{
  "scores": {{
    "technical": <0-10>,
    "ambiguity": <0-10>,
    "realism": <0-10>,
    "difficulty": <0-10>,
    "distractors": <0-10>
  }},
  "issues": ["<specific problem>", "..."]
}}

Scoring guidance:
- technical: is the GCP/ML content and the marked correct answer actually correct?
- ambiguity: could more than one option be defensibly correct? (higher = less ambiguous)
- realism: does it read like a real PMLE scenario?
- difficulty: does it require multi-step reasoning rather than recall?
- distractors: are the wrong options plausible yet clearly incorrect to an expert?"""

_JUDGE_SYSTEM = """\
You are the quality judge. You do not rewrite questions; you decide.
Base your decision STRICTLY on the SEARCH EVIDENCE and the critic report. Do not use
your own prior knowledge to justify correctness. If the evidence does not verify the
marked correct answer, do not ACCEPT.
Return ONLY valid JSON. No prose, no markdown fences."""

_JUDGE_USER = """\
Decide whether this question is exam-ready using ONLY the search evidence.

SEARCH EVIDENCE (from DuckDuckGo, official GCP docs):
---
{evidence}
---

QUESTION JSON:
{question_json}

DETERMINISTIC ISSUES (must all be resolved to accept):
{deterministic_issues}

CRITIC REPORT:
{critic_json}

Return ONLY this JSON:
{{
  "overall_score": <0-10 float>,
  "decision": "ACCEPT" | "IMPROVE" | "REJECT",
  "critical_issues": ["<blocking problem>", "..."],
  "improvement_instructions": ["<precise, actionable fix>", "..."]
}}"""

_IMPROVER_SYSTEM = """\
You are an exam-item improver. Rewrite the SAME question to fix the listed problems
while keeping the topic and intent. Ground every fact and the correct answer STRICTLY
in the SEARCH EVIDENCE provided; do not invent details from prior knowledge.
Return ONLY a single valid JSON object matching the input schema (id, topic,
question_type, answer_mode, case_study_context, question, choices, correct_answer,
explanation, reference). No prose, no markdown fences."""

_IMPROVER_USER = """\
Rewrite this question to resolve every issue, grounded ONLY in the search evidence.

SEARCH EVIDENCE (from DuckDuckGo, official GCP docs):
---
{evidence}
---

ORIGINAL QUESTION JSON:
{question_json}

DETERMINISTIC ISSUES:
{deterministic_issues}

JUDGE CRITICAL ISSUES:
{critical_issues}

IMPROVEMENT INSTRUCTIONS:
{improvement_instructions}

Return ONLY the improved question as a single JSON object."""


def _run_json_chain(system: str, user: str, variables: dict[str, Any], temperature: float, fast: bool = False) -> Any:
    llm = get_llm(temperature=temperature, fast=fast)
    chain = (
        ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        | llm
        | StrOutputParser()
    )
    return _parse_json(chain.invoke(variables), fallback={})


# ── Nodes ────────────────────────────────────────────────────────────────────

def generate_node(state: QuestionState) -> dict[str, Any]:
    raw = _invoke_generation(state["ctx"], num_questions=1, temperature=0.3)
    question = normalize_one(raw[0], idx=1) if raw else None
    if question is None:
        return {"question": {}, "iteration": 0, "final_status": "REJECT"}
    return {"question": question, "iteration": 0}


def deterministic_node(state: QuestionState) -> dict[str, Any]:
    return {"deterministic_issues": validate_question(state.get("question", {}))}


def gather_verification_evidence(ctx: dict[str, Any], question: dict[str, Any]) -> str:
    """Search DuckDuckGo for evidence that grounds the validation agents."""
    topic = ctx.get("topic", {})
    exam_title = ctx.get("exam_title", "")
    choices = question.get("choices", {})
    correct = question.get("correct_answer", [])
    answer_texts = " ".join(choices.get(k, "") for k in correct)
    query_fragment = f"{topic.get('name', '')} {answer_texts}".strip()

    try:
        results = search_topic(exam_title, query_fragment)
    except Exception:
        results = []

    parts = [
        f"[{r.get('title', '')}] {r.get('url', '')}\n{r.get('body', '')}"
        for r in results[:5]
        if r.get("body")
    ]
    return "\n\n".join(parts) or "No search evidence found."


def research_node(state: QuestionState) -> dict[str, Any]:
    evidence = gather_verification_evidence(state.get("ctx", {}), state.get("question", {}))
    return {"evidence": evidence}


def critic_node(state: QuestionState) -> dict[str, Any]:
    import json

    critic = _run_json_chain(
        _CRITIC_SYSTEM,
        _CRITIC_USER,
        {
            "evidence": state.get("evidence", "No search evidence found."),
            "question_json": json.dumps(state.get("question", {}), ensure_ascii=False),
        },
        temperature=0,
        fast=True,
    )
    return {"critic": critic if isinstance(critic, dict) else {}}


def judge_node(state: QuestionState) -> dict[str, Any]:
    import json

    det_issues = state.get("deterministic_issues", [])
    judge = _run_json_chain(
        _JUDGE_SYSTEM,
        _JUDGE_USER,
        {
            "evidence": state.get("evidence", "No search evidence found."),
            "question_json": json.dumps(state.get("question", {}), ensure_ascii=False),
            "deterministic_issues": json.dumps(det_issues, ensure_ascii=False),
            "critic_json": json.dumps(state.get("critic", {}), ensure_ascii=False),
        },
        temperature=0,
    )
    if not isinstance(judge, dict):
        judge = {}

    try:
        score = float(judge.get("overall_score", 0))
    except (TypeError, ValueError):
        score = 0.0
    critical = judge.get("critical_issues") or []
    iteration = state.get("iteration", 0)

    final_status = ""
    if not det_issues and not critical and score >= QUESTION_ACCEPT_SCORE:
        final_status = "ACCEPT"
    elif iteration >= QUESTION_MAX_ITERATIONS:
        final_status = "REJECT"

    result: dict[str, Any] = {"judge": judge}
    if final_status:
        result["final_status"] = final_status
    return result


def improver_node(state: QuestionState) -> dict[str, Any]:
    import json

    judge = state.get("judge", {})
    improved_raw = _run_json_chain(
        _IMPROVER_SYSTEM,
        _IMPROVER_USER,
        {
            "evidence": state.get("evidence", "No search evidence found."),
            "question_json": json.dumps(state.get("question", {}), ensure_ascii=False),
            "deterministic_issues": json.dumps(state.get("deterministic_issues", []), ensure_ascii=False),
            "critical_issues": json.dumps(judge.get("critical_issues", []), ensure_ascii=False),
            "improvement_instructions": json.dumps(judge.get("improvement_instructions", []), ensure_ascii=False),
        },
        temperature=0.3,
    )
    improved = normalize_one(improved_raw, idx=state.get("question", {}).get("id", 1))
    question = improved if improved is not None else state.get("question", {})
    return {"question": question, "iteration": state.get("iteration", 0) + 1}



# ── Routing ──────────────────────────────────────────────────────────────────

def _route_after_generate(state: QuestionState) -> str:
    return "reject" if state.get("final_status") == "REJECT" else "validate"


def _route_after_judge(state: QuestionState) -> str:
    return "final" if state.get("final_status") else "improve"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_question_graph():
    graph = StateGraph(QuestionState)
    graph.add_node("generate", generate_node)
    graph.add_node("deterministic", deterministic_node)
    graph.add_node("research", research_node)
    graph.add_node("critic", critic_node)
    graph.add_node("judge", judge_node)
    graph.add_node("improver", improver_node)

    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", _route_after_generate, {"validate": "deterministic", "reject": END})
    graph.add_edge("deterministic", "research")
    graph.add_edge("research", "critic")
    graph.add_edge("critic", "judge")
    graph.add_conditional_edges("judge", _route_after_judge, {"improve": "improver", "final": END})
    graph.add_edge("improver", "deterministic")

    return graph.compile()


_COMPILED_GRAPH = None


def run_question_pipeline(ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Run the critic-refiner graph for one question. Returns the accepted question or None."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_question_graph()

    final_state = _COMPILED_GRAPH.invoke({"ctx": ctx})
    if final_state.get("final_status") != "ACCEPT":
        return None
    question = final_state.get("question")
    if not question or validate_question(question):
        return None
    return question
