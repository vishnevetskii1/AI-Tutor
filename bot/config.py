# Роль файла: Загружает runtime-конфиг из переменных окружения.
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Config:
    telegram_token: str
    openrouter_api_key: str
    llm_provider: str
    llm_model: str
    ollama_model: str
    embedding_model: str
    ollama_url: str
    ollama_timeout_seconds: float
    qdrant_url: str
    qdrant_timeout_seconds: float
    sqlite_path: str
    disciplines_dir: str
    admin_user_ids: list[int]
    enable_remote_ops: bool
    lesson_viewer_base_url: str
    lesson_viewer_bind: str
    lesson_viewer_port: int


def load_config() -> Config:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    if provider == "openrouter" and not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY не задан в .env")

    enable_remote_ops = os.getenv("ENABLE_REMOTE_OPS", "false").lower() in {
        "1", "true", "yes", "on"
    }
    admin_user_ids = [
        int(value.strip())
        for value in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",")
        if value.strip()
    ]

    return Config(
        telegram_token=token,
        openrouter_api_key=openrouter_api_key,
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_timeout_seconds=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "5")),
        sqlite_path=os.getenv("SQLITE_PATH", "data/tutor.db"),
        disciplines_dir=os.getenv("DISCIPLINES_DIR", "data/disciplines"),
        admin_user_ids=admin_user_ids,
        enable_remote_ops=enable_remote_ops,
        lesson_viewer_base_url=os.getenv("LESSON_VIEWER_BASE_URL", "").rstrip("/"),
        lesson_viewer_bind=os.getenv("LESSON_VIEWER_BIND", "0.0.0.0"),
        lesson_viewer_port=int(os.getenv("LESSON_VIEWER_PORT", "8080")),
    )
