# GCP PMLE Quizzer

An AI-powered exam quiz generator for the **GCP Professional Machine Learning Engineer** certification. Upload your exam guide and study materials, and the app generates scenario-based practice questions grounded in official GCP documentation and your own study materials.

![Python](https://img.shields.io/badge/python-3.12-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.58-red) ![LangChain](https://img.shields.io/badge/langchain-0.3-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Features

- **Dual AI backend** — switch between Claude (Anthropic) and Ollama (local) in one `.env` line
- **RAG grounding** — FAISS vector store indexes all your uploaded study materials; every question is grounded in your own content
- **Web search grounding** — DuckDuckGo search (free, no API key) fetches current official GCP documentation for each topic
- **Strict item writing** — questions follow professional psychometric standards: Bloom's Taxonomy levels 3–6, scenario-based stems, 4 choices, no negation in the stem, no "all of the above"
- **Topic-aware generation** — AI extracts exam objectives from your uploaded guide; you choose which topics to practice
- **Interactive quiz** — one question at a time, sidebar navigator, back/forward navigation, answers persist
- **Results review** — score by topic, per-choice explanations, optional live link validation for reference URLs

---

## Architecture

```
gcp_pmlc_quizzer/
├── app.py                       # Home page + backend health check
├── pages/
│   ├── 1_Upload_Document.py     # Upload + parse + RAG index
│   ├── 2_Configure_Quiz.py      # Topic select + question generation
│   ├── 3_Take_Quiz.py           # Interactive quiz interface
│   └── 4_Review_Results.py      # Score, explanations, reference links
├── utils/
│   ├── ai_client.py             # LangChain factory (Claude / Ollama)
│   ├── rag_engine.py            # FAISS vector store + retrieval
│   ├── document_parser.py       # PDF / DOCX / TXT parsing
│   ├── question_generator.py    # LangChain chains + prompt engineering
│   ├── web_search.py            # DuckDuckGo search + page fetcher
│   ├── session_manager.py       # Streamlit session state management
│   └── link_validator.py        # Concurrent HTTP link validation
└── config/
    └── settings.py              # All settings loaded from .env
```

**Generation pipeline per topic:**

```
DuckDuckGo search → fetch official docs
        +
FAISS similarity search → retrieve study material chunks
        +
LangChain chain (system: Item Writing Guidelines)
        ↓
Exam-quality MCQ questions (JSON)
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- One of:
  - **Claude**: An [Anthropic API key](https://console.anthropic.com/)
  - **Ollama**: [Ollama](https://ollama.com/) running locally with a model pulled

### Installation

```bash
git clone https://github.com/singaravelan/gcp-pmlc-quizzer.git
cd gcp-pmlc-quizzer

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` for your chosen backend:

**Claude:**
```env
AI_BACKEND=claude
ANTHROPIC_API_KEY=sk-ant-...
```

**Ollama:**
```env
AI_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Pull Ollama models if needed:
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Usage

1. **Upload Document** — Upload your GCP PMLE exam guide (PDF, DOCX, or TXT). Optionally add extra study materials (whitepapers, notes). The app builds a FAISS RAG index from all files.

2. **Configure Quiz** — Select which exam topics to practice, set questions per topic (1–10), and choose a Bloom's Taxonomy level (Application → Evaluation).

3. **Take Quiz** — Answer one question at a time. Use the sidebar navigator to jump between questions. Your answers are saved automatically.

4. **Review Results** — See your score broken down by topic, read per-choice explanations, and follow verified links to the official GCP documentation that grounded each question.

---

## AI Backends

| Backend | Model recommendation | Notes |
|---|---|---|
| Claude | `claude-sonnet-4-6` | Best question quality; requires API key |
| Ollama | `qwen2.5:7b` (8 GB RAM) | Free, local, good JSON adherence |
| Ollama | `qwen2.5:14b` (16 GB RAM) | Better reasoning, recommended if RAM allows |
| Ollama | `llama3.1:8b` | Solid alternative to qwen |

Embeddings:
- **Ollama backend**: uses `nomic-embed-text` (768-dim, fast)
- **Claude backend**: uses `all-MiniLM-L6-v2` via HuggingFace (~80 MB, downloaded once)

---

## Item Writing Standards

Every generated question follows professional psychometric guidelines:

- Bloom's Taxonomy levels 3–6 only (no recall or definition questions)
- Real-world GCP scenario in every stem
- No negation ("not") in the question stem
- Exactly 4 choices — one unambiguously correct, three plausible distractors
- No "all of the above" or "none of the above"
- Parallel answer structure
- Every question grounded in fetched official documentation + RAG context
- Reference URL cited per question

---

## License

MIT
