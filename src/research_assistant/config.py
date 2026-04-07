from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

_DEFAULT_DATASETS_DIR = Path(__file__).parent / "eval" / "datasets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_model: str = "anthropic:claude-haiku-4-5-20251001"
    anthropic_api_key: SecretStr | None = None

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    qdrant_mode: Literal["memory", "local", "server", "cloud"] = "local"
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    collection_name: str = "documents"

    chunk_max_tokens: int = 512

    top_k: int = 3
    max_tokens: int = 512

    cache_dir: Path = Path(".cache/edgar")

    datasets_dir: Path = _DEFAULT_DATASETS_DIR

    logfire_token: SecretStr | None = None
    log_level: str = "INFO"

    eval_judge_model: str = "anthropic:claude-haiku-4-5-20251001"
    openai_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_logfire(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    import logfire

    token = settings.logfire_token.get_secret_value() if settings.logfire_token else None
    logfire.configure(token=token)
    logfire.instrument_pydantic_ai()
