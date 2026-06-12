"""
Question generation pipeline using LangChain chains.

Three stages:
  1. classify_document()   — is this an exam-related document?
  2. extract_topics()      — extract exam objectives and knowledge areas
  3. generate_questions()  — create MCQ questions grounded in RAG + web sources
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.ai_client import get_llm
from config.settings import BLOOM_LEVELS


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

COGNITIVE LEVEL:
- Use ONLY Bloom's Taxonomy levels 3 (Application), 4 (Analysis), 5 (Synthesis), \
or 6 (Evaluation).
- Every question MUST present a realistic professional scenario that requires the \
candidate to apply knowledge to solve a problem, evaluate a solution, or make a \
design decision.
- Never write recall or definition questions.

STEM (QUESTION TEXT):
- Write in active voice, present tense, 6th-grade reading level.
- The stem must NOT contain the word "not" or any negation.
- Each question must be fully self-contained (no "refer to the exhibit" style).
- Avoid idioms, humor, slang, and trick questions.
- Make the question independent of all other questions in the set.

ANSWER CHOICES:
- Provide EXACTLY 4 choices labeled A, B, C, D.
- There must be exactly ONE unambiguously correct answer.
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
Generate exactly {num_questions} multiple-choice questions for the topic below.

EXAM: {exam_title}
TOPIC: {topic_name}
TOPIC DESCRIPTION: {topic_description}
SUBTOPICS TO COVER: {subtopics}
BLOOM'S LEVEL: {bloom_level_name} (Level {bloom_level})

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
    "bloom_level": {bloom_level},
    "bloom_level_name": "{bloom_level_name}",
    "question": "<complete scenario-based question stem as a full sentence ending with ?>",
    "choices": {{
      "A": "<choice A text>",
      "B": "<choice B text>",
      "C": "<choice C text>",
      "D": "<choice D text>"
    }},
    "correct_answer": "A",
    "explanation": "<2-4 sentences: why the correct answer is right, and why each \
distractor is wrong — be specific, cite concepts from the source documentation>",
    "reference": "<one URL from the AVAILABLE SOURCE URLS above>"
  }}
]"""


def generate_questions(
    exam_title: str,
    topic: dict[str, Any],
    num_questions: int,
    bloom_level: int,
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
        bloom_level: Bloom's taxonomy level (3–6).
        rag_context: Retrieved chunks from the user's uploaded study materials.
        web_context: Fetched content from official documentation sources.
        source_urls: List of real URLs to use in question references.

    Returns:
        List of question dicts conforming to the question JSON schema.
    """
    bloom_name = BLOOM_LEVELS.get(bloom_level, "Application")
    subtopics_str = ", ".join(topic.get("subtopics", [topic.get("name", "")]))
    urls_str = "\n".join(source_urls or []) or "No verified URLs available."

    llm = get_llm(temperature=0.3)
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
        "exam_title": exam_title,
        "topic_name": topic.get("name", ""),
        "topic_description": topic.get("description", ""),
        "subtopics": subtopics_str,
        "bloom_level": bloom_level,
        "bloom_level_name": bloom_name,
        "web_context": web_context[:12_000] if web_context else "No web content retrieved.",
        "rag_context": rag_context[:8_000] if rag_context else "No study material context retrieved.",
        "source_urls": urls_str,
    })

    result = _parse_json(raw, fallback=[])
    if not isinstance(result, list):
        return []
    return result


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
