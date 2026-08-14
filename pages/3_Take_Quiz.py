"""
Page 3 — Take Quiz

Displays one question at a time with:
  - Progress bar and question counter
  - Radio-button answer selection (unique key per question)
  - Previous/Next navigation (answers persist across navigation)
  - Submit action on the final question
"""
import streamlit as st

from config.settings import APP_TITLE
from utils.session_manager import (
    init_session,
    get,
    set,
    reset_quiz,
    has_questions,
    quiz_complete,
    SS_QUESTIONS,
    SS_CURRENT_Q_IDX,
    SS_ANSWERS,
    SS_QUIZ_COMPLETE,
)

st.set_page_config(
    page_title=f"Take Quiz — {APP_TITLE}",
    page_icon="🎯",
    layout="centered",
)
init_session()

st.title("🎯 Quiz")

# ── Guards ────────────────────────────────────────────────────────────────────
if not has_questions():
    st.warning("No quiz has been generated yet. Please configure a quiz first.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️ Configure Quiz", use_container_width=True):
            st.switch_page("pages/2_Configure_Quiz.py")
    with col2:
        if st.button("📄 Upload Document", use_container_width=True):
            st.switch_page("pages/1_Upload_Document.py")
    st.stop()

if quiz_complete():
    st.success("You have completed the quiz!")
    if st.button("📊 View Results", type="primary", use_container_width=True):
        st.switch_page("pages/4_Review_Results.py")
    if st.button("🔄 Retake Quiz", use_container_width=True):
        reset_quiz()
        st.rerun()
    st.stop()

# ── Load state ────────────────────────────────────────────────────────────────
questions = get(SS_QUESTIONS, [])
idx = get(SS_CURRENT_Q_IDX, 0)
total = len(questions)
answers = get(SS_ANSWERS, {})

# Clamp idx to valid range
idx = max(0, min(idx, total - 1))
q = questions[idx]

# ── Progress ──────────────────────────────────────────────────────────────────
answered_count = len(answers)
st.progress(
    answered_count / total,
    text=f"Question {idx + 1} of {total} | {answered_count} answered",
)

st.divider()

# ── Question display ──────────────────────────────────────────────────────────
topic_label = q.get("topic", "")
question_type = q.get("question_type", "single_answer").replace("_", " ").title()
answer_mode = q.get("answer_mode", "single")
case_study_context = q.get("case_study_context", "")

st.caption(f"Topic: {topic_label} | Type: {question_type}")

if case_study_context:
    st.markdown("#### Case Study Context")
    st.info(case_study_context)

st.markdown(f"### Q{idx + 1}. {q['question']}")
st.markdown("")

# Build ordered answer choices
choices: dict[str, str] = q.get("choices", {})
choice_keys = sorted(choices.keys())  # A, B, C, D
choice_labels = [f"**{k}.**  {choices[k]}" for k in choice_keys]

# Determine if already answered
already_answered = q["id"] in answers
prior_answer = answers.get(q["id"])

selected_keys: list[str] = []
if answer_mode == "multi":
    # Multi-select questions require exactly two answers.
    prior_multi = prior_answer if isinstance(prior_answer, list) else []
    selected_labels = st.multiselect(
        "Select exactly two answers:",
        options=choice_labels,
        default=[f"**{k}.**  {choices[k]}" for k in prior_multi if k in choices],
        key=f"q_multi_{q['id']}",
        disabled=already_answered,
    )
    selected_keys = [lbl.split(".")[0].replace("**", "").strip() for lbl in selected_labels]
else:
    # Map prior answer key to label index for pre-selection
    if isinstance(prior_answer, list) and prior_answer and prior_answer[0] in choice_keys:
        default_idx = choice_keys.index(prior_answer[0])
    else:
        default_idx = None

    selected_label = st.radio(
        "Select your answer:",
        options=choice_labels,
        index=default_idx,
        key=f"q_radio_{q['id']}",
        disabled=already_answered,
        label_visibility="collapsed",
    )
    if selected_label:
        selected_keys = [selected_label.split(".")[0].replace("**", "").strip()]

# ── Answer feedback (when already answered) ───────────────────────────────────
if already_answered:
    correct_keys = q.get("correct_answer", [])
    user_keys = prior_answer if isinstance(prior_answer, list) else []

    if set(user_keys) == set(correct_keys):
        st.success(f"You selected **{', '.join(user_keys)}** — Correct!")
    else:
        st.error(
            f"You selected **{', '.join(user_keys) if user_keys else 'none'}** — "
            f"The correct answer is **{', '.join(correct_keys)}**."
        )
    st.info("Full explanations available on the Results page after submitting.")

st.divider()

# ── Navigation buttons ────────────────────────────────────────────────────────
nav_left, nav_center, nav_right = st.columns([1, 2, 1])

with nav_left:
    if idx > 0:
        if st.button("← Previous", use_container_width=True):
            set(SS_CURRENT_Q_IDX, idx - 1)
            st.rerun()

with nav_center:
    # Show unanswered count as a hint
    unanswered = total - answered_count
    if unanswered > 0:
        st.caption(f"{unanswered} question(s) remaining")
    else:
        st.caption("All questions answered — ready to submit!")

with nav_right:
    if not already_answered:
        # Submit current answer
        is_last = idx == total - 1
        btn_label = "Submit & Finish" if is_last else "Submit & Next →"
        can_submit = bool(selected_keys)
        if answer_mode == "multi":
            can_submit = len(selected_keys) == 2
            if not can_submit:
                st.caption("Select exactly two answers for this question.")

        if st.button(btn_label, type="primary", use_container_width=True, disabled=not can_submit):
            answers[q["id"]] = selected_keys
            set(SS_ANSWERS, answers)

            if is_last:
                set(SS_QUIZ_COMPLETE, True)
                st.rerun()
            else:
                set(SS_CURRENT_Q_IDX, idx + 1)
                st.rerun()
    else:
        # Already answered — just navigate
        is_last = idx == total - 1
        if is_last:
            if answered_count == total:
                if st.button("Finish & View Results →", type="primary", use_container_width=True):
                    set(SS_QUIZ_COMPLETE, True)
                    st.switch_page("pages/4_Review_Results.py")
            else:
                st.caption("Answer remaining questions to finish.")
        else:
            if st.button("Next →", type="primary", use_container_width=True):
                set(SS_CURRENT_Q_IDX, idx + 1)
                st.rerun()

# ── Question navigator (sidebar) ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Question Navigator")
    st.markdown("Click a number to jump to that question.")

    cols_per_row = 5
    q_rows = [questions[i:i+cols_per_row] for i in range(0, total, cols_per_row)]

    for row in q_rows:
        btn_cols = st.columns(len(row))
        for j, rq in enumerate(row):
            q_num = questions.index(rq) + 1
            rq_idx = q_num - 1
            rq_id = rq["id"]
            is_current = rq_idx == idx
            is_answered = rq_id in answers

            label = str(q_num)
            if is_current:
                label = f"[{q_num}]"

            btn_type = "primary" if is_current else "secondary"

            with btn_cols[j]:
                if st.button(label, key=f"nav_{rq_id}", use_container_width=True, type=btn_type):
                    set(SS_CURRENT_Q_IDX, rq_idx)
                    st.rerun()

    st.divider()
    st.markdown(f"**Answered:** {answered_count}/{total}")

    correct_so_far = sum(
        1 for question in questions
        if set(answers.get(question["id"], [])) == set(question.get("correct_answer", []))
    )
    if answered_count > 0:
        st.markdown(f"**Correct so far:** {correct_so_far}/{answered_count}")

    if answered_count == total:
        st.divider()
        if st.button("Submit & View Results", type="primary", use_container_width=True):
            set(SS_QUIZ_COMPLETE, True)
            st.switch_page("pages/4_Review_Results.py")
