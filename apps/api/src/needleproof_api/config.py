from __future__ import annotations

from ipaddress import ip_network
from pathlib import Path

from pydantic import Field, model_validator
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
    max_opened_tokens: int = 50_000
    max_document_inspections: int = 2
    max_inspection_pages: int = 3
    max_inspection_characters: int = 24_000
    soft_timeout_seconds: float = 45.0
    hard_timeout_seconds: float = 60.0
    trace_include_sensitive_data: bool = False
    receipt_retention_days: int = 7
    max_concurrent_runs: int = 5
    public_demo: bool = False
    session_cookie_name: str = "needleproof_session"
    session_cookie_secure: bool = False
    trusted_proxy_cidrs: list[str] = Field(default_factory=list)
    max_runs_per_session_per_hour: int = 20
    max_runs_per_ip_per_hour: int = 60
    max_model_tokens_per_hour: int = 1_000_000
    max_model_tokens_per_day: int = 5_000_000
    model_token_reservation_per_run: int = 100_000

    @model_validator(mode="after")
    def require_secure_cookie_for_public_demo(self) -> Settings:
        if self.public_demo and not self.session_cookie_secure:
            raise ValueError("Public demo mode requires a Secure browser-session cookie")
        if self.model_token_reservation_per_run <= 0:
            raise ValueError("Model-token reservation per run must be positive")
        if self.model_token_reservation_per_run > min(
            self.max_model_tokens_per_hour, self.max_model_tokens_per_day
        ):
            raise ValueError(
                "Model-token reservation per run cannot exceed the hourly or daily budget"
            )
        try:
            for cidr in self.trusted_proxy_cidrs:
                ip_network(cidr, strict=False)
        except ValueError as exc:
            raise ValueError(f"Invalid trusted proxy CIDR: {exc}") from exc
        return self

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
