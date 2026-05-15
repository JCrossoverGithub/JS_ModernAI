"""
config.py — Centralised configuration using pydantic-settings.

All environment variables are declared here with typed defaults.
Use `get_settings()` (cached) rather than importing `os.getenv` directly.

Usage:
    from config import get_settings
    settings = get_settings()
    print(settings.qdrant_host)
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- FastAPI / CORS ---
    cors_origin: str = "http://localhost:5000"
    upload_dir: str = "./uploads"

    # --- Embeddings ---
    embed_device: str = "cuda"
    hf_home: str = "/data/hf_cache"

    # --- Qdrant (Phase 3+) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "library_ai"
    qdrant_memory_collection: str = "chat_memory"

    # --- MongoDB (Phase 2+) ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "llm_twin"

    # --- Ollama (Phase 1 legacy, replaced in Phase 7) ---
    ollama_base_url: str = "http://localhost:11434"

    # --- HuggingFace (Phase 4+) ---
    huggingface_access_token: str = ""
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    finetuned_model_id: str = ""

    # --- Inference mode ---
    # "ollama"       → current local Ollama server (Phase 1)
    # "huggingface"  → HF pipeline with local / fine-tuned weights (Phase 7)
    # "sagemaker"    → AWS SageMaker endpoint (Phase 9)
    inference_mode: str = "ollama"

    # --- Comet ML / Opik monitoring (Phase 11) ---
    comet_api_key: str = ""
    comet_project_name: str = "llm-twin"
    opik_api_key: str = ""
    opik_project_name: str = "llm-twin"

    # --- ZenML (Phase 2+) ---
    zenml_store_url: str = ""   # empty → local default stack


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
