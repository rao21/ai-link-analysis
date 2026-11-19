from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-6-luna",
}


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables / `.env`.

    API keys are `SecretStr` so they never show up in logs, reprs or tracebacks.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    llm_model: str = ""
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_effort: Literal["low", "medium", "high"] = "low"
    llm_timeout_s: float = 60
    max_output_tokens: int = 8000

    # Website fetching
    fetch_timeout_s: float = 10
    max_html_bytes: int = 2_000_000
    max_redirects: int = 5
    max_subpages: int = 3
    max_context_chars: int = 24_000
    min_content_chars: int = 300

    # Cost and abuse protection
    cache_ttl_s: int = 21_600
    rate_limit: str = "10/minute"
    # True only when running behind a proxy that APPENDS the client IP to X-Forwarded-For
    # (Render, Heroku, most load balancers). Otherwise the header is client-controlled.
    trust_proxy_headers: bool = False
    daily_budget_usd: float = 2.0


    @property
    def resolved_model(self) -> str:
        """`LLM_MODEL` if set, otherwise the recommended default for the provider."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
