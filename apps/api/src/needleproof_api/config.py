from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEEDLEPROOF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = "gpt-5.6-terra"
    reasoning_effort: str = "medium"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 768
    data_dir: Path = Path("data")
    corpus_source: Path = Path("data/corpus-source.json")
    top_k: int = 8
    max_top_k: int = 20
    max_turns: int = 6
    max_tool_calls: int = 12
    max_searches: int = 4
    max_opened_chunks: int = 50
    max_document_inspections: int = 2
    soft_timeout_seconds: float = 45.0
    hard_timeout_seconds: float = 60.0
    trace_include_sensitive_data: bool = False
    receipt_retention_days: int = 7
    max_concurrent_runs: int = 5

    @property
    def app_db_path(self) -> Path:
        return self.data_dir / "needleproof.sqlite3"

    @property
    def corpora_dir(self) -> Path:
        return self.data_dir / "corpora"

    @property
    def receipts_dir(self) -> Path:
        return self.data_dir / "receipts"

    @property
    def rehearsal_path(self) -> Path:
        return self.data_dir / "rehearsal" / "featured.json"


settings = Settings()
