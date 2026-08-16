"""
Page 4 — Review Results

Displays:
  - Score summary with pass/warn/fail feedback
  - Per-question review with correct/incorrect highlighting
  - Detailed per-choice explanations
  - Verified source reference links (optional HTTP validation)
  - Options to retake or start a new quiz
"""
import streamlit as st

from config.settings import APP_TITLE
from utils.session_manager import (
    init_session,
    get,
    set,
    reset_quiz,
    reset_all,
    has_questions,
    quiz_complete,
    SS_QUESTIONS,
    SS_ANSWERS,
    SS_EXAM_TITLE,
    SS_QUIZ_COMPLETE,
)
from utils.link_validator import validate_links_batch

st.set_page_config(
    page_title=f"Results — {APP_TITLE}",
    page_icon="📊",
    layout="wide",
)
init_session()

st.title("📊 Quiz Results")

# ── Guards ────────────────────────────────────────────────────────────────────
questions = get(SS_QUESTIONS, [])
answers = get(SS_ANSWERS, {})

if not questions:
    st.warning("No quiz has been completed yet.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️ Configure Quiz", use_container_width=True):
            st.switch_page("pages/2_Configure_Quiz.py")
    with col2:
        if st.button("📄 Upload Document", use_container_width=True):
            st.switch_page("pages/1_Upload_Document.py")
    st.stop()

if not get(SS_QUIZ_COMPLETE) and not answers:
    st.warning("Quiz is not yet complete. Please finish all questions first.")
    if st.button("🎯 Go to Quiz", type="primary", use_container_width=True):
        st.switch_page("pages/3_Take_Quiz.py")
    st.stop()

# ── Score Calculation ─────────────────────────────────────────────────────────
total = len(questions)


def is_correct(q: dict) -> bool:
    return set(answers.get(q["id"], [])) == set(q.get("correct_answer", []))


correct_count = sum(
    1 for q in questions if is_correct(q)
)
incorrect_count = total - correct_count
score_pct = (correct_count / total * 100) if total > 0 else 0
exam_title = get(SS_EXAM_TITLE, "Exam")

# ── Score Header ──────────────────────────────────────────────────────────────
st.subheader(f"Score: {correct_count}/{total} ({score_pct:.0f}%)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Questions", total)
m2.metric("Correct", correct_count, delta=None)
m3.metric("Incorrect", incorrect_count)
m4.metric("Score", f"{score_pct:.0f}%")

if score_pct >= 80:
    st.success(
        "Excellent performance! You demonstrate strong command of this material. "
        "You are on track for the certification exam."
    )
elif score_pct >= 60:
    st.warning(
        "Good effort. Review the explanations for incorrect answers carefully. "
        "Focus on the topics where you missed questions."
    )
else:
    st.error(
        "Keep studying. Work through the detailed explanations below and revisit "
        "the official GCP documentation for each topic you missed."
    )

# ── Topic Breakdown ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Performance by Topic")

topic_stats: dict[str, dict] = {}
for q in questions:
    topic = q.get("topic", "Unknown")
    if topic not in topic_stats:
        topic_stats[topic] = {"correct": 0, "total": 0}
    topic_stats[topic]["total"] += 1
    if is_correct(q):
        topic_stats[topic]["correct"] += 1

topic_cols = st.columns(min(4, len(topic_stats)))
for i, (topic, stats) in enumerate(topic_stats.items()):
    pct = stats["correct"] / stats["total"] * 100
    with topic_cols[i % len(topic_cols)]:
        st.metric(
            label=topic[:30] + ("..." if len(topic) > 30 else ""),
            value=f"{stats['correct']}/{stats['total']}",
            delta=f"{pct:.0f}%",
            delta_color="normal" if pct >= 60 else "inverse",
        )

# ── Review Controls ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Question Review")

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
with ctrl_col1:
    show_explanations = st.toggle("Show explanations", value=True)
with ctrl_col2:
    filter_mode = st.selectbox(
        "Filter questions",
        options=["All questions", "Incorrect only", "Correct only"],
        index=0,
    )
with ctrl_col3:
    validate_links = st.checkbox(
        "Validate reference links",
        value=False,
        help="Checks each reference URL for accessibility (slower load)",
    )

# ── Link Validation (opt-in) ──────────────────────────────────────────────────
link_status: dict[str, tuple[bool, int]] = {}
if validate_links:
    all_refs = [
        q.get("reference", "")
        for q in questions
        if q.get("reference", "").startswith("http")
    ]
    if all_refs:
        with st.spinner(f"Validating {len(all_refs)} reference links..."):
            link_status = validate_links_batch(all_refs)

# ── Filter questions ──────────────────────────────────────────────────────────
if filter_mode == "Incorrect only":
    display_questions = [q for q in questions if not is_correct(q)]
elif filter_mode == "Correct only":
    display_questions = [q for q in questions if is_correct(q)]
else:
    display_questions = questions

if not display_questions:
    st.info("No questions match the current filter.")
else:
    # ── Per-question Review ───────────────────────────────────────────────────
    for q in display_questions:
        q_id = q["id"]
        user_ans = answers.get(q_id, [])
        correct_ans = q.get("correct_answer", [])
        correct = is_correct(q)

        icon = "✅" if correct else "❌"
        topic_tag = q.get("topic", "")
        qtype_tag = q.get("question_type", "single_answer").replace("_", " ").title()
        answer_mode = q.get("answer_mode", "single")
        case_study_context = q.get("case_study_context", "")

        # Auto-expand incorrect questions
        with st.expander(
            f"{icon}  Q{q_id}: {q['question'][:90]}{'...' if len(q['question']) > 90 else ''}",
            expanded=not correct,
        ):
            st.caption(f"Topic: {topic_tag} | Type: {qtype_tag}")
            if answer_mode == "multi":
                st.caption("This is a multiple-select question (exactly two correct answers).")
            if case_study_context:
                st.markdown("**Case Study Context**")
                st.info(case_study_context)
            st.markdown(f"**{q['question']}**")
            st.markdown("")

            # Render answer choices with highlighting
            choices = q.get("choices", {})
            for key in sorted(choices.keys()):
                choice_text = choices[key]
                prefix = ""
                suffix = ""

                if key in correct_ans:
                    prefix = "✅ "
                    suffix = " ← **Correct Answer**"
                elif key in user_ans and not correct:
                    prefix = "❌ "
                    suffix = " ← *Your Answer*"
                elif key in user_ans and correct:
                    prefix = "✅ "
                    suffix = " ← *Your Answer*"

                st.markdown(f"{prefix}**{key}.** {choice_text}{suffix}")

            # Answer summary
            st.markdown("")
            if correct:
                st.success(f"Your answer: **{', '.join(user_ans)}** — Correct!")
            else:
                if not user_ans:
                    st.warning(f"Not answered | Correct answer: **{', '.join(correct_ans)}**")
                else:
                    st.error(
                        f"Your answer: **{', '.join(user_ans)}** | "
                        f"Correct answer: **{', '.join(correct_ans)}**"
                    )

            # Explanation
            if show_explanations:
                explanation = q.get("explanation", "")
                if explanation:
                    st.markdown("---")
                    st.markdown("**Explanation:**")
                    st.info(explanation)

            # Reference link
            ref = q.get("reference", "")
            if ref:
                st.markdown("---")
                if ref.startswith("http") and validate_links and ref in link_status:
                    valid, code = link_status[ref]
                    if valid:
                        st.markdown(f"**Reference:** [View Official Documentation]({ref}) ✅")
                    else:
                        st.markdown(
                            f"**Reference:** ~~{ref}~~ "
                            f"(link unavailable — HTTP {code})"
                        )
                elif ref.startswith("http"):
                    st.markdown(f"**Reference:** [View Official Documentation]({ref})")
                else:
                    st.markdown(f"**Reference:** {ref}")

# ── Action Buttons ────────────────────────────────────────────────────────────
st.divider()
action1, action2, action3 = st.columns(3)

with action1:
    if st.button("🔄 Retake Same Quiz", use_container_width=True, type="primary"):
        reset_quiz()
        st.switch_page("pages/3_Take_Quiz.py")

with action2:
    if st.button("⚙️ New Quiz Configuration", use_container_width=True):
        reset_quiz()
        st.switch_page("pages/2_Configure_Quiz.py")

with action3:
    if st.button("🔁 Start Over (New Document)", use_container_width=True):
        reset_all()
        st.switch_page("pages/1_Upload_Document.py")
