"""
Page 2 — Configure Quiz

Lets the user:
  1. Select which exam topics to include
  2. Set the number of questions per topic
  3. Choose Bloom's taxonomy level
  4. Generate the full question set (with web search + RAG grounding)
"""
import random

import streamlit as st

from config.settings import APP_TITLE, BLOOM_LEVELS
from utils.session_manager import (
    init_session,
    get,
    set,
    reset_quiz,
    has_document,
    has_topics,
    SS_TOPICS,
    SS_EXAM_TITLE,
    SS_SELECTED_TOPICS,
    SS_NUM_QUESTIONS,
    SS_BLOOM_LEVEL,
    SS_QUESTIONS,
    SS_VECTOR_STORE,
)
from utils.question_generator import generate_questions
from utils.web_search import gather_source_content
from utils.rag_engine import retrieve_context

st.set_page_config(
    page_title=f"Configure Quiz — {APP_TITLE}",
    page_icon="⚙️",
    layout="wide",
)
init_session()

st.title("⚙️ Configure Your Quiz")

# ── Guard: need a document with topics ───────────────────────────────────────
if not has_document():
    st.warning("No document loaded. Please upload an exam guide first.")
    if st.button("📄 Go to Upload"):
        st.switch_page("pages/1_Upload_Document.py")
    st.stop()

if not has_topics():
    st.warning("Topics could not be extracted. Please re-upload your document.")
    if st.button("📄 Go to Upload"):
        st.switch_page("pages/1_Upload_Document.py")
    st.stop()

topics = get(SS_TOPICS, [])
exam_title = get(SS_EXAM_TITLE, "Unknown Exam")
vector_store = get(SS_VECTOR_STORE)

st.subheader(f"Exam: {exam_title}")
st.divider()

# ── Section 1: Topic Selection ────────────────────────────────────────────────
st.subheader("1. Select Topics")
st.markdown(
    "Choose the topics you want practice questions for. "
    "The app will search official GCP documentation and your study materials for each selected topic."
)

all_topic_names = [t["name"] for t in topics]
default_selection = all_topic_names[:min(3, len(all_topic_names))]

selected_names = st.multiselect(
    "Topics to include in your quiz:",
    options=all_topic_names,
    default=default_selection,
    help="Select at least 1 topic. More topics = more questions and longer generation time.",
)

selected_topic_ids = [t["id"] for t in topics if t["name"] in selected_names]
set(SS_SELECTED_TOPICS, selected_topic_ids)

if not selected_names:
    st.warning("Select at least one topic to continue.")
    st.stop()

# Show subtopics for selected topics
with st.expander("View subtopics for selected topics", expanded=False):
    for topic in topics:
        if topic["name"] in selected_names:
            subtopics = topic.get("subtopics", [])
            st.markdown(f"**{topic['name']}**: {', '.join(subtopics) if subtopics else 'No subtopics listed'}")

st.divider()

# ── Section 2: Quiz Parameters ────────────────────────────────────────────────
st.subheader("2. Quiz Parameters")

param_col1, param_col2 = st.columns(2)

with param_col1:
    num_q = st.slider(
        "Questions per topic",
        min_value=1,
        max_value=10,
        value=get(SS_NUM_QUESTIONS, 5),
        help="How many questions to generate for each selected topic.",
    )
    set(SS_NUM_QUESTIONS, num_q)

with param_col2:
    bloom_name_to_level = {v: k for k, v in BLOOM_LEVELS.items()}
    bloom_options = list(BLOOM_LEVELS.values())

    current_bloom = get(SS_BLOOM_LEVEL, 4)
    current_bloom_name = BLOOM_LEVELS.get(current_bloom, "Analysis")
    default_idx = bloom_options.index(current_bloom_name) if current_bloom_name in bloom_options else 1

    bloom_name = st.selectbox(
        "Bloom's Taxonomy Level",
        options=bloom_options,
        index=default_idx,
        help=(
            "Application (L3): Apply knowledge to solve problems\n"
            "Analysis (L4): Break down and examine components\n"
            "Synthesis (L5): Combine concepts to design solutions\n"
            "Evaluation (L6): Judge and justify design decisions"
        ),
    )
    bloom_level = bloom_name_to_level[bloom_name]
    set(SS_BLOOM_LEVEL, bloom_level)

total_questions = len(selected_topic_ids) * num_q

st.info(
    f"**Summary:** {len(selected_topic_ids)} topic(s) × {num_q} question(s) = "
    f"**{total_questions} total questions** at **{bloom_name}** level (Bloom's L{bloom_level})"
)

st.divider()

# ── Section 3: Generate ───────────────────────────────────────────────────────
st.subheader("3. Generate Questions")
st.markdown(
    "For each topic, the app will:\n"
    "1. Search official GCP documentation (DuckDuckGo, free)\n"
    "2. Retrieve relevant chunks from your uploaded study materials (RAG)\n"
    "3. Generate scenario-based exam questions grounded in both sources"
)

if vector_store is None:
    st.warning(
        "RAG knowledge base is not available. "
        "Questions will be generated using web search only. "
        "Re-upload your documents to enable RAG grounding."
    )

if st.button("Generate Quiz", type="primary", use_container_width=True):
    reset_quiz()
    selected_topics = [t for t in topics if t["id"] in selected_topic_ids]
    all_questions: list[dict] = []

    progress_bar = st.progress(0.0, text="Starting question generation...")
    generation_errors: list[str] = []

    for i, topic in enumerate(selected_topics):
        topic_name = topic["name"]
        progress_pct = i / len(selected_topics)
        progress_bar.progress(
            progress_pct,
            text=f"Generating questions for: **{topic_name}** ({i+1}/{len(selected_topics)})...",
        )

        # Step A: Web search for official documentation
        with st.spinner(f"Searching official docs for '{topic_name}'..."):
            try:
                web_context, source_urls = gather_source_content(exam_title, topic)
            except Exception as e:
                web_context, source_urls = "", []
                st.warning(f"Web search failed for '{topic_name}': {e}")

        # Step B: RAG retrieval from uploaded study materials
        rag_context = ""
        if vector_store is not None:
            try:
                query = f"{exam_title} {topic_name} {' '.join(topic.get('subtopics', []))}"
                rag_context = retrieve_context(query, vector_store)
            except Exception as e:
                st.warning(f"RAG retrieval failed for '{topic_name}': {e}")

        # Step C: Generate questions
        try:
            qs = generate_questions(
                exam_title=exam_title,
                topic=topic,
                num_questions=num_q,
                bloom_level=bloom_level,
                rag_context=rag_context,
                web_context=web_context,
                source_urls=source_urls,
            )

            # Assign globally unique IDs
            for q in qs:
                q["id"] = len(all_questions) + 1
                q["topic"] = topic_name

            all_questions.extend(qs)
            st.success(
                f"Generated {len(qs)} question(s) for **{topic_name}** "
                f"({len(source_urls)} source URL(s) found)"
            )
        except Exception as e:
            generation_errors.append(f"{topic_name}: {e}")
            st.error(f"Failed to generate questions for '{topic_name}': {e}")

    progress_bar.progress(1.0, text="Question generation complete!")

    if all_questions:
        # Randomize order
        random.shuffle(all_questions)
        # Re-assign sequential IDs after shuffle
        for idx, q in enumerate(all_questions, start=1):
            q["id"] = idx

        set(SS_QUESTIONS, all_questions)
        st.success(f"**{len(all_questions)} questions ready!** Click below to start the quiz.")
        st.divider()
        if st.button("Start Quiz →", type="primary", use_container_width=True):
            st.switch_page("pages/3_Take_Quiz.py")
    else:
        st.error(
            "No questions were generated. Check your AI backend connection and try again. "
            "You may also try selecting fewer topics or a different Bloom's level."
        )
        if generation_errors:
            with st.expander("Error details"):
                for err in generation_errors:
                    st.code(err)

# ── Show existing questions if already generated ──────────────────────────────
existing_questions = get(SS_QUESTIONS, [])
if existing_questions and not st.session_state.get("_generating"):
    st.divider()
    st.info(
        f"**{len(existing_questions)} questions** already generated. "
        "Click **Generate Quiz** above to regenerate, or proceed to the quiz."
    )
    if st.button("Continue to Quiz →", type="secondary", use_container_width=True):
        st.switch_page("pages/3_Take_Quiz.py")
