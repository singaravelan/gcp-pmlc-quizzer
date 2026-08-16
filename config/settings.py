import os

# Fix SSL PermissionError due to environment keylog file settings in sandbox environments
if "SSLKEYLOGFILE" in os.environ:
    try:
        with open(os.environ["SSLKEYLOGFILE"], "a") as f:
            pass
    except Exception:
        os.environ.pop("SSLKEYLOGFILE", None)

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# Do not override existing process/container environment variables with .env file values
load_dotenv(BASE_DIR / ".env", override=False)

# AI Backend selection
AI_BACKEND: str = os.getenv("AI_BACKEND", "claude")  # "claude" | "ollama"

# Claude settings
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MODEL_FAST: str = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-4-5-20251001")

# Ollama settings
is_docker = os.path.exists("/.dockerenv") or os.getenv("RUNNING_IN_DOCKER") == "true"
default_ollama_url = "http://host.docker.internal:11434" if is_docker else "http://localhost:11434"
raw_ollama_url = os.getenv("OLLAMA_BASE_URL", default_ollama_url)

# Automatically rewrite localhost / 127.0.0.1 to host.docker.internal when inside a Docker container
if is_docker:
    raw_ollama_url = raw_ollama_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

OLLAMA_BASE_URL: str = raw_ollama_url
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Embeddings backend when AI_BACKEND=claude: "huggingface" | "ollama"
EMBED_BACKEND: str = os.getenv("EMBED_BACKEND", "huggingface")
HUGGINGFACE_EMBED_MODEL: str = "all-MiniLM-L6-v2"

# App settings
APP_TITLE: str = os.getenv("APP_TITLE", "GCP PMLE Quizzer")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}
CACHE_DIR: Path = BASE_DIR / "data" / "cache"

# RAG settings
RAG_CHUNK_SIZE: int = 1000
RAG_CHUNK_OVERLAP: int = 200
RAG_TOP_K: int = 5

# Web search settings
SEARCH_NUM_RESULTS: int = 5
SEARCH_FETCH_TOP_N: int = 3
SEARCH_PAGE_MAX_CHARS: int = 8000
SEARCH_DELAY_SECONDS: float = 0.5

# Question generation graph (critic-refiner workflow)
QUESTION_MAX_ITERATIONS: int = int(os.getenv("QUESTION_MAX_ITERATIONS", "3"))
QUESTION_ACCEPT_SCORE: float = float(os.getenv("QUESTION_ACCEPT_SCORE", "8.5"))

# Link validation
LINK_VALIDATE_TIMEOUT: int = 5
LINK_VALIDATE_MAX_WORKERS: int = 8
