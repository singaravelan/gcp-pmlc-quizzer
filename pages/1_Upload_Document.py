"""
Page 1 — Upload Document

Flow:
  1. User uploads exam guide + optional study materials (PDF/DOCX/TXT)
  2. Documents are parsed to plain text
  3. AI classifies the primary document (is it exam-related?)
  4. AI extracts exam topics from the primary document
  5. All documents are chunked and indexed into a FAISS vector store
  6. Everything is stored in session_state for downstream pages
"""
import streamlit as st

from config.settings import APP_TITLE, SUPPORTED_EXTENSIONS
from utils.session_manager import (
    init_session,
    get,
    set,
    reset_all,
    has_document,
    has_topics,
    SS_DOCUMENT_TEXT,
    SS_DOCUMENT_NAME,
    SS_STUDY_DOCS,
    SS_VECTOR_STORE,
    SS_IS_EXAM_DOC,
    SS_EXAM_TITLE,
    SS_TOPICS,
)
from utils.document_parser import parse_document, validate_file_size, truncate_for_context
from utils.question_generator import classify_document, extract_topics
from utils.rag_engine import build_vector_store, count_chunks

st.set_page_config(
    page_title=f"Upload Document — {APP_TITLE}",
    page_icon="📄",
    layout="wide",
)
init_session()

st.title("📄 Upload Exam Document")
st.markdown(
    "Upload your GCP PMLE exam guide and any additional study materials. "
    "The app will extract topics and build a knowledge base for question generation."
)
st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("1. Select Files")
st.markdown(
    "- **First file** = your primary exam guide (used for topic extraction)\n"
    "- **Additional files** = study notes, whitepapers, or reference docs (added to RAG index)"
)

uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Supported: PDF, DOCX, TXT | Max 20 MB per file",
)

if uploaded_files:
    st.info(f"**{len(uploaded_files)}** file(s) selected: {', '.join(f.name for f in uploaded_files)}")

    if st.button("Analyze and Index Documents", type="primary", use_container_width=True):
        reset_all()

        all_texts: list[str] = []
        all_names: list[str] = []
        parse_errors: list[str] = []

        with st.status("Processing documents...", expanded=True) as status:

            # ── Step 1: Parse all files ───────────────────────────────────────
            st.write("**Step 1:** Parsing uploaded files...")
            for i, f in enumerate(uploaded_files):
                size_err = validate_file_size(f.size, f.name)
                if size_err:
                    parse_errors.append(size_err)
                    st.warning(f"Skipping {f.name}: {size_err}")
                    continue

                try:
                    text = parse_document(f.read(), f.name)
                    all_texts.append(text)
                    all_names.append(f.name)
                    st.write(f"  ✅ Parsed **{f.name}** ({len(text):,} characters)")
                except Exception as e:
                    parse_errors.append(f"{f.name}: {e}")
                    st.warning(f"  ❌ Could not parse **{f.name}**: {e}")

            if not all_texts:
                status.update(label="No files could be parsed", state="error")
                st.error("None of the uploaded files could be parsed. Please try different files.")
                st.stop()

            # Primary document is the first successfully parsed file
            primary_text = all_texts[0]
            primary_name = all_names[0]
            set(SS_DOCUMENT_TEXT, primary_text)
            set(SS_DOCUMENT_NAME, primary_name)

            # Store additional study docs
            study_docs = [
                {"filename": name, "text": text}
                for name, text in zip(all_names[1:], all_texts[1:])
            ]
            set(SS_STUDY_DOCS, study_docs)

            # ── Step 2: Classify primary document ────────────────────────────
            st.write("**Step 2:** Classifying primary document...")
            classification = classify_document(primary_text)
            is_exam = classification.get("is_exam_related", True)
            set(SS_IS_EXAM_DOC, is_exam)

            if is_exam:
                st.write(
                    f"  ✅ Exam document confirmed "
                    f"(confidence: {classification.get('confidence', 'unknown')}) — "
                    f"{classification.get('reason', '')}"
                )
            else:
                st.warning(
                    f"  ⚠️ **{primary_name}** may not be an exam guide "
                    f"({classification.get('reason', '')}). "
                    f"Proceeding anyway — select only relevant topics in Step 2."
                )

            # ── Step 3: Extract topics ────────────────────────────────────────
            st.write("**Step 3:** Extracting exam topics...")
            topic_data = extract_topics(truncate_for_context(primary_text))
            topics = topic_data.get("topics", [])
            exam_title = topic_data.get("exam_title", "Unknown Exam")

            if not topics:
                status.update(label="Topic extraction failed", state="error")
                st.error(
                    "Could not extract topics from this document. "
                    "Try uploading a more detailed exam guide or syllabus."
                )
                st.stop()

            set(SS_EXAM_TITLE, exam_title)
            set(SS_TOPICS, topics)
            st.write(f"  ✅ Extracted **{len(topics)}** topics for **{exam_title}**")

            # ── Step 4: Build FAISS vector store ─────────────────────────────
            st.write("**Step 4:** Building RAG knowledge base (this may take a moment)...")
            try:
                vector_store = build_vector_store(all_texts, all_names)
                set(SS_VECTOR_STORE, vector_store)
                chunk_count = count_chunks(vector_store)
                st.write(
                    f"  ✅ Indexed **{len(all_texts)}** document(s) → "
                    f"**{chunk_count:,}** chunks in FAISS vector store"
                )
            except Exception as e:
                st.warning(
                    f"  ⚠️ RAG indexing failed: {e}. "
                    f"Questions will be generated without RAG context."
                )

            status.update(label="Documents analyzed successfully!", state="complete")

# ── Display current state ────────────────────────────────────────────────────
if has_topics():
    st.divider()
    exam_title = get(SS_EXAM_TITLE)
    topics = get(SS_TOPICS, [])
    vector_store = get(SS_VECTOR_STORE)
    study_docs = get(SS_STUDY_DOCS, [])

    st.subheader(f"Exam: {exam_title}")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Topics Detected", len(topics))
    m2.metric(
        "RAG Chunks",
        count_chunks(vector_store) if vector_store else 0,
        help="Number of text chunks indexed for retrieval",
    )
    m3.metric("Primary Doc", get(SS_DOCUMENT_NAME, "—"))
    m4.metric("Extra Study Docs", len(study_docs))

    st.divider()
    st.subheader("Detected Topics")
    st.markdown("These topics were extracted from your exam guide. You'll select which ones to include in Step 2.")

    # Display topics in a 3-column grid
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            with st.expander(f"**{topic['name']}**", expanded=False):
                st.markdown(topic.get("description", "No description available."))
                subtopics = topic.get("subtopics", [])
                if subtopics:
                    st.markdown("**Key subtopics:**")
                    for sub in subtopics:
                        st.markdown(f"- {sub}")

    st.divider()
    if st.button("Configure Quiz →", type="primary", use_container_width=True):
        st.switch_page("pages/2_Configure_Quiz.py")

elif has_document():
    st.warning("Topics could not be extracted. Try uploading a different document.")
else:
    st.divider()
    st.markdown("### Getting Started")
    st.markdown(
        "Upload the **GCP Professional Machine Learning Engineer** exam guide or study guide above. "
        "You can also add supplementary materials such as:\n"
        "- Google Cloud documentation exports\n"
        "- Whitepapers (ML on GCP, MLOps, etc.)\n"
        "- Your own study notes\n\n"
        "The app will use all uploaded materials to generate grounded exam-quality questions."
    )
