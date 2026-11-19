"""Real API calls. Run with: pytest -m live   (skipped for providers without a key in .env)."""

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.services.llm.base import get_llm_client

pytestmark = pytest.mark.live


class Tiny(BaseModel):
    company: str
    services: list[str]


@pytest.mark.parametrize("provider", ["gemini", "anthropic", "openai"])
async def test_structured_output_roundtrip(provider):
    settings = Settings(llm_provider=provider)
    key = {"gemini": settings.gemini_api_key, "anthropic": settings.anthropic_api_key, "openai": settings.openai_api_key}[provider]
    if key is None or not key.get_secret_value():
        pytest.skip(f"no {provider} key configured")
    client = get_llm_client(settings)
    result = await client.generate_json(
        "Extract the company name and its services as JSON.",
        "Acme Digital builds websites and runs SEO audits in Lahore.",
        Tiny,
    )
    assert Tiny.model_validate(result.data).company
    assert result.input_tokens > 0 and result.output_tokens > 0
