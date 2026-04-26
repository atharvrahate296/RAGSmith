"""
RAGSmith – Centralised configuration
All settings are read from environment variables (or a .env file).
This is the SINGLE SOURCE OF TRUTH for every configurable value.

Local dev  → copy .env.example to .env, set LLM_PROVIDER=ollama, DB_DRIVER=sqlite
AWS deploy → set env vars on EC2 or in systemd unit, LLM_PROVIDER=groq, DB_DRIVER=postgres
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    log_level: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────────────
    # "sqlite"   → uses local file at data/ragsmith22.db  (default, no setup needed)
    # "postgres" → uses DATABASE_URL (required on AWS)
    db_driver: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = ""          # e.g. postgresql://user:pass@host:5432/ragsmith
    sqlite_path: str = "data/ragsmith22.db"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # "ollama" → local Ollama instance (default for local dev)
    # "groq"   → Groq cloud API        (default for AWS)
    llm_provider: Literal["ollama", "groq"] = "ollama"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "mistral:7b"      # Exact model name from Ollama
    ollama_available_models: str = "mistral:7b,gemma:2b,llama2:7b,neural-chat:7b"  # Comma-separated list

    # Groq settings
    groq_api_key: str = ""
    groq_default_model: str = "mixtral-8x7b-32768"  # Exact Groq model name
    groq_available_models: str = "mixtral-8x7b-32768,llama2-70b-4096,gemma-7b-it"  # Comma-separated list

    # ── Embedding model ───────────────────────────────────────────────────────
    # Always a local SentenceTransformer — never changes between environments
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── File Storage ──────────────────────────────────────────────────────────
    # "local" → data/uploads/ directory on disk
    # "s3"    → AWS S3 bucket
    storage_backend: Literal["local", "s3"] = "local"
    local_upload_dir: str = "data/uploads"

    # S3 settings (only needed when storage_backend=s3)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    # ── FAISS / Vector indexes ────────────────────────────────────────────────
    # Always stored on local disk (EBS volume on EC2)
    faiss_index_dir: str = "data/indexes"
    faiss_chunks_dir: str = "data/chunks"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1          # Keep at 1 — FAISS index is in-process, not shared
    reload: bool = False      # Set True only in development

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_mb: int = 50

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, or "*" for all
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def effective_llm_model(self) -> str:
        """Return the default model name for the active LLM provider."""
        if self.llm_provider == "groq":
            return self.groq_default_model
        return self.ollama_default_model


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance — import and call this everywhere.

        from config import get_settings
        cfg = get_settings()
    """
    return Settings()
