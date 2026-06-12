"""
GCP PMLE Quizzer — Home Page

Entry point for the Streamlit multi-page application.
Displays backend health status and navigation shortcuts.
"""
import streamlit as st

from config.settings import APP_TITLE, AI_BACKEND, CLAUDE_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL
from utils.session_manager import (
    init_session,
    has_document,
    has_topics,
    has_questions,
    quiz_complete,
    get,
    SS_EXAM_TITLE,
    SS_QUESTIONS,
    SS_ANSWERS,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()

# ── Header ───────────────────────────────────────────────────────────────────
st.title(f"🎓 {APP_TITLE}")
st.markdown(
    "**AI-powered exam quiz generator grounded in official GCP documentation and your study materials.**"
)
st.divider()

# ── Session status banner ────────────────────────────────────────────────────
col_status, col_help = st.columns([1, 1])

with col_status:
    st.subheader("Session Status")

    if has_document():
        exam_title = get(SS_EXAM_TITLE) or "Exam"
        st.success(f"Document loaded: **{exam_title}**")
    else:
        st.info("No document loaded yet.")

    if has_topics():
        from utils.session_manager import SS_TOPICS
        topics = get(SS_TOPICS, [])
        st.success(f"{len(topics)} topics detected.")
    else:
        st.info("No topics extracted yet.")

    if has_questions():
        questions = get(SS_QUESTIONS, [])
        answers = get(SS_ANSWERS, {})
        answered = len(answers)
        total = len(questions)
        if quiz_complete():
            score = sum(
                1 for q in questions
                if answers.get(q["id"]) == q.get("correct_answer")
            )
            st.success(f"Quiz complete — Score: {score}/{total} ({score/total*100:.0f}%)")
        else:
            st.warning(f"Quiz in progress — {answered}/{total} answered.")
    else:
        st.info("No quiz generated yet.")

with col_help:
    st.subheader("How to Use")
    st.markdown("""
    **Step 1 — Upload Document**
    Upload your GCP PMLE exam guide (PDF, DOCX, or TXT). Optionally add extra
    study materials. The app builds a RAG index from all uploaded files.

    **Step 2 — Configure Quiz**
    Select the exam topics you want to practice, set the number of questions,
    and choose the cognitive difficulty level.

    **Step 3 — Take Quiz**
    Answer AI-generated scenario-based questions one at a time. Navigate
    freely — your answers are saved automatically.

    **Step 4 — Review Results**
    See your score, detailed explanations for each question, and verified
    links to the official GCP documentation used to ground each question.
    """)

st.divider()

# ── Backend Health Check ──────────────────────────────────────────────────────
st.subheader("AI Backend Status")

bcol1, bcol2 = st.columns(2)

with bcol1:
    from utils.ai_client import get_backend_display_name, get_embed_display_name
    backend_name = get_backend_display_name()
    embed_name = get_embed_display_name()

    st.markdown(f"**LLM Backend:** {backend_name}")
    st.markdown(f"**Embeddings:** {embed_name}")

    if st.button("Check Backend Connectivity", use_container_width=True):
        with st.spinner("Testing LLM connection..."):
            from utils.ai_client import check_llm_availability
            ok, err = check_llm_availability()
        if ok:
            st.success("LLM is reachable and responding.")
        else:
            st.error(f"LLM connection failed: {err}")
            if AI_BACKEND == "ollama":
                st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")
            else:
                st.info("Check that ANTHROPIC_API_KEY is set correctly in your .env file.")

with bcol2:
    st.markdown("**Switch backends** by editing `.env`:")
    st.code(
        "# For Claude:\nAI_BACKEND=claude\nANTHROPIC_API_KEY=sk-ant-...\n\n"
        "# For Ollama:\nAI_BACKEND=ollama\nOLLAMA_MODEL=llama3.2\n"
        "OLLAMA_EMBED_MODEL=nomic-embed-text",
        language="bash",
    )

st.divider()

# ── Quick Navigation ──────────────────────────────────────────────────────────
st.subheader("Quick Navigation")
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("### 📄 Upload")
    st.markdown("Upload exam guide and study materials")
    if st.button("Go to Upload", key="nav_upload", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Upload_Document.py")

with nav2:
    st.markdown("### ⚙️ Configure")
    st.markdown("Select topics and generate questions")
    btn_type = "primary" if has_document() else "secondary"
    if st.button("Go to Configure", key="nav_config", use_container_width=True, type=btn_type):
        if not has_document():
            st.warning("Upload a document first.")
        else:
            st.switch_page("pages/2_Configure_Quiz.py")

with nav3:
    st.markdown("### 🎯 Quiz")
    st.markdown("Take the generated quiz")
    btn_type = "primary" if has_questions() else "secondary"
    if st.button("Go to Quiz", key="nav_quiz", use_container_width=True, type=btn_type):
        if not has_questions():
            st.warning("Generate a quiz first.")
        else:
            st.switch_page("pages/3_Take_Quiz.py")

with nav4:
    st.markdown("### 📊 Results")
    st.markdown("Review answers and explanations")
    btn_type = "primary" if quiz_complete() else "secondary"
    if st.button("Go to Results", key="nav_results", use_container_width=True, type=btn_type):
        if not quiz_complete():
            st.warning("Complete the quiz first.")
        else:
            st.switch_page("pages/4_Review_Results.py")
