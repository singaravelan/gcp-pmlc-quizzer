"""
Question generation pipeline using LangChain chains.

Three stages:
  1. classify_document()   — is this an exam-related document?
  2. extract_topics()      — extract exam objectives and knowledge areas
    3. generate_questions()  — create PMLE-style questions grounded in RAG + web sources
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.ai_client import get_llm


# ── Stage 1: Document Classification ────────────────────────────────────────

_CLASSIFY_SYSTEM = """\
You are a document classifier for a professional certification exam preparation tool.
Your output must be valid JSON only. Do not include markdown, prose, or code fences."""

_CLASSIFY_USER = """\
Determine whether the document excerpt below is related to a professional \
certification exam, study guide, syllabus, or technical examination outline.

Respond with ONLY this JSON (no other text):
{{
  "is_exam_related": true,
  "confidence": "high",
  "reason": "One sentence explaining your classification."
}}

Document excerpt:
---
{excerpt}
---"""


def classify_document(text: str) -> dict[str, Any]:
    """
    Classify whether the document is exam/study-guide related.

    Returns dict with: is_exam_related (bool), confidence (str), reason (str).
    """
    llm = get_llm(temperature=0, fast=True)
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", _CLASSIFY_SYSTEM),
            ("human", _CLASSIFY_USER),
        ])
        | llm
        | StrOutputParser()
    )
    raw = chain.invoke({"excerpt": text[:3000]})
    return _parse_json(raw, fallback={"is_exam_related": True, "confidence": "low", "reason": "Parse failed"})


# ── Stage 2: Topic Extraction ────────────────────────────────────────────────

_TOPIC_SYSTEM = """\
You are an expert certification exam analyst. Extract the key knowledge domains, \
learning objectives, and topic areas from the provided exam document.
Your output must be valid JSON only. Do not include markdown or code fences."""

_TOPIC_USER = """\
Extract all major exam topics, knowledge areas, and learning objectives \
from the document below.

Return ONLY this JSON structure (no other text):
{{
  "exam_title": "<inferred exam title>",
  "topics": [
    {{
      "id": 1,
      "name": "<topic name>",
      "description": "<one-sentence description of what this topic covers>",
      "subtopics": ["<subtopic 1>", "<subtopic 2>", "<subtopic 3>"]
    }}
  ]
}}

Guidelines:
- Include 5 to 15 high-level topics (no more).
- Each topic should represent a distinct knowledge domain tested in the exam.
- Subtopics should be specific skills or concepts within that domain.
- If the document is for GCP Professional Machine Learning Engineer, use the official exam domains.

Document:
---
{document_text}
---"""


def extract_topics(document_text: str) -> dict[str, Any]:
    """
    Extract exam topics and knowledge areas from document text.

    Returns dict with: exam_title (str), topics (list of dicts).
    """
    llm = get_llm(temperature=0)
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", _TOPIC_SYSTEM),
            ("human", _TOPIC_USER),
        ])
        | llm
        | StrOutputParser()
    )
    raw = chain.invoke({"document_text": document_text[:60_000]})
    return _parse_json(raw, fallback={"exam_title": "Unknown Exam", "topics": []})


# ── Stage 3: Question Generation ────────────────────────────────────────────

_QUESTION_SYSTEM = """\
You are a senior psychometrician and professional exam item writer specializing in \
cloud computing and machine learning certifications.

ITEM WRITING GUIDELINES — follow these strictly for every question:

PMLE EXAM PATTERN:
- Build questions in the real PMLE style: scenario-driven, constraint-heavy, architecture \
trade-offs, lifecycle ordering, diagnostics, optimization, and metric interpretation.
- Include a mix of these question types across each generated set when possible:
    - single_answer
    - multiple_select
    - case_study
    - scenario_architecture
    - best_solution
    - first_step
    - next_step
    - service_mapping
    - troubleshooting
    - optimization
    - constraint_heavy
    - metrics_interpretation
- For "multiple_select", use stems like "Which TWO actions should you take?" and return \
exactly two correct choices.
- For "case_study", provide reusable scenario context in "case_study_context" and keep \
the question stem focused on a single decision.
- Prefer "best" answer framing when multiple options are plausible, selecting the option \
that best satisfies stated constraints with the least operational burden.
- Never write recall-only or definition-only questions.

STEM (QUESTION TEXT):
- Write in active voice, present tense, 6th-grade reading level.
- The stem must NOT contain the word "not" or any negation.
- Each question must be fully self-contained (no "refer to the exhibit" style).
- Avoid idioms, humor, slang, and trick questions.
- Make the question independent of all other questions in the set.

ANSWER CHOICES:
- Provide EXACTLY 4 choices labeled A, B, C, D.
- For single-answer styles: exactly ONE correct answer.
- For multiple-select styles: exactly TWO correct answers.
- The 3 distractors must be plausible but clearly incorrect to someone with deep \
knowledge of the subject.
- Choices must be parallel in structure and similar in length.
- NEVER use "All of the above", "None of the above", or similar constructs.
- Vary the position of the correct answer across the question set.

GROUNDING:
- Base every question on the SOURCE DOCUMENTATION and RAG CONTEXT provided.
- The correct answer must be verifiable against the provided source material.
- Cite one real URL from the provided source URLs in the "reference" field.

OUTPUT FORMAT:
- Return ONLY a valid JSON array. No prose, no markdown fences, no other text.
- Each element must match the schema provided in the user message exactly."""

_QUESTION_USER = """\
Generate exactly {num_questions} PMLE-style exam questions for the topic below.

EXAM: {exam_title}
TOPIC: {topic_name}
TOPIC DESCRIPTION: {topic_description}
SUBTOPICS TO COVER: {subtopics}

SOURCE DOCUMENTATION (ground your questions in this content):
---
{web_context}
---

RAG CONTEXT FROM STUDY MATERIALS (additional grounding):
---
{rag_context}
---

AVAILABLE SOURCE URLS (use one per question in the "reference" field):
{source_urls}

Return ONLY a JSON array with exactly {num_questions} items. Each item must follow \
this schema exactly:
[
  {{
    "id": 1,
    "topic": "{topic_name}",
        "question_type": "single_answer",
        "answer_mode": "single",
        "case_study_context": "",
    "question": "<complete scenario-based question stem as a full sentence ending with ?>",
    "choices": {{
      "A": "<choice A text>",
      "B": "<choice B text>",
      "C": "<choice C text>",
      "D": "<choice D text>"
    }},
        "correct_answer": ["A"],
    "explanation": "<2-4 sentences: why the correct answer is right, and why each \
distractor is wrong — be specific, cite concepts from the source documentation>",
    "reference": "<one URL from the AVAILABLE SOURCE URLS above>"
  }}
]

Hard requirements:
- Use answer_mode="single" with a one-item correct_answer array for single-answer styles.
- Use answer_mode="multi" with exactly two correct_answer keys for multiple-select.
- If question_type="case_study", include a non-empty case_study_context.
- Ensure each correct_answer key exists in choices.
- Keep reference as a real URL from AVAILABLE SOURCE URLS when possible.
"""


def _invoke_generation(
    ctx: dict[str, Any],
    num_questions: int,
    temperature: float = 0.3,
) -> list[dict[str, Any]]:
    """Invoke the generation chain and return the raw parsed question list."""
    topic = ctx.get("topic", {})
    subtopics_str = ", ".join(topic.get("subtopics", [topic.get("name", "")]))
    urls_str = "\n".join(ctx.get("source_urls") or []) or "No verified URLs available."
    web_context = ctx.get("web_context", "")
    rag_context = ctx.get("rag_context", "")

    llm = get_llm(temperature=temperature)
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", _QUESTION_SYSTEM),
            ("human", _QUESTION_USER),
        ])
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke({
        "num_questions": num_questions,
        "exam_title": ctx.get("exam_title", ""),
        "topic_name": topic.get("name", ""),
        "topic_description": topic.get("description", ""),
        "subtopics": subtopics_str,
        "web_context": web_context[:12_000] if web_context else "No web content retrieved.",
        "rag_context": rag_context[:8_000] if rag_context else "No study material context retrieved.",
        "source_urls": urls_str,
    })

    result = _parse_json(raw, fallback=[])
    return result if isinstance(result, list) else []


def generate_questions(
    exam_title: str,
    topic: dict[str, Any],
    num_questions: int,
    rag_context: str = "",
    web_context: str = "",
    source_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate exam-quality MCQ questions for a single topic.

    Args:
        exam_title: Name of the certification exam.
        topic: Dict with keys: name, description, subtopics.
        num_questions: Number of questions to generate.
        rag_context: Retrieved chunks from the user's uploaded study materials.
        web_context: Fetched content from official documentation sources.
        source_urls: List of real URLs to use in question references.

    Returns:
        List of question dicts conforming to the question JSON schema.
    """
    ctx = {
        "exam_title": exam_title,
        "topic": topic,
        "rag_context": rag_context,
        "web_context": web_context,
        "source_urls": source_urls or [],
    }

    from utils.question_graph import run_question_pipeline

    results: list[dict[str, Any]] = []
    for _ in range(num_questions):
        question = run_question_pipeline(ctx)
        if question is not None:
            results.append(question)
    return results


CHOICE_KEYS = {"A", "B", "C", "D"}


def normalize_one(q: Any, idx: int = 1) -> dict[str, Any] | None:
    """Coerce a single raw LLM question into the stable schema, or None if unusable."""
    if not isinstance(q, dict):
        return None

    choices = q.get("choices")
    if not isinstance(choices, dict):
        return None

    clean_choices = {
        k: str(v).strip()
        for k, v in choices.items()
        if k in CHOICE_KEYS and str(v).strip()
    }

    answer_mode = str(q.get("answer_mode", "single")).lower().strip()
    answer_mode = "multi" if answer_mode == "multi" else "single"

    raw_correct = q.get("correct_answer", [])
    if isinstance(raw_correct, str):
        correct = [raw_correct.strip().upper()]
    elif isinstance(raw_correct, list):
        correct = [str(x).strip().upper() for x in raw_correct]
    else:
        correct = []
    correct = list(dict.fromkeys(k for k in correct if k in CHOICE_KEYS))

    return {
        "id": int(q.get("id", idx)) if str(q.get("id", idx)).isdigit() else idx,
        "topic": str(q.get("topic", "")).strip(),
        "question_type": str(q.get("question_type", "single_answer")).strip() or "single_answer",
        "answer_mode": answer_mode,
        "case_study_context": str(q.get("case_study_context", "")).strip(),
        "question": str(q.get("question", "")).strip(),
        "choices": clean_choices,
        "correct_answer": correct,
        "explanation": str(q.get("explanation", "")).strip(),
        "reference": str(q.get("reference", "")).strip(),
    }


def validate_question(q: dict[str, Any]) -> list[str]:
    """Return deterministic rule violations for a normalized question (empty = pass)."""
    issues: list[str] = []
    choices = q.get("choices", {})
    correct = q.get("correct_answer", [])
    mode = q.get("answer_mode", "single")

    if set(choices.keys()) != CHOICE_KEYS:
        issues.append("Choices must be exactly four options labeled A, B, C, D.")

    if not correct:
        issues.append("No correct answer provided.")
    for key in correct:
        if key not in choices:
            issues.append(f"Correct answer '{key}' is not among the choices.")

    if mode == "multi" and len(correct) != 2:
        issues.append("Multiple-select questions must have exactly two correct answers.")
    if mode == "single" and len(correct) != 1:
        issues.append("Single-answer questions must have exactly one correct answer.")

    texts = [t.strip().lower() for t in choices.values()]
    if len(texts) != len(set(texts)):
        issues.append("Choices contain duplicate option text.")

    if not q.get("question"):
        issues.append("Question stem is empty.")
    if not q.get("explanation"):
        issues.append("Explanation is empty.")

    stem = q.get("question", "").lower()
    for key in correct:
        answer_text = choices.get(key, "").strip().lower()
        if answer_text and answer_text in stem:
            issues.append("Correct answer text is leaked in the question stem.")
            break

    if q.get("question_type") == "case_study" and not q.get("case_study_context"):
        issues.append("Case-study questions must include case_study_context.")

    return issues


# ── JSON Parsing Helper ──────────────────────────────────────────────────────

def _parse_json(raw: str, fallback: Any) -> Any:
    """
    Robustly parse a JSON string that may be wrapped in markdown code fences.
    Ollama models almost always add fences; Claude sometimes does too.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost JSON object or array
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    return fallback
