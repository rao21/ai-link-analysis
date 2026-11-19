import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.errors import LLMRefusedError, LLMUnavailableError
from app.pricing import PRICES_PER_MTOK, Price, estimate_cost
from app.schemas import Competitor, LLMAnalysis, SearchQuery
from app.services.llm.base import get_llm_client, strict_json_schema

FIXTURES = Path(__file__).parent / "fixtures"


def llm_fixture() -> dict:
    return json.loads((FIXTURES / "llm_valid.json").read_text())


# ---------- schemas ----------

def test_llm_fixture_validates():
    analysis = LLMAnalysis.model_validate(llm_fixture())
    assert analysis.brand.name == "Acme Digital"


def test_no_volume_fields_in_schema():
    schema_text = json.dumps(LLMAnalysis.model_json_schema()).lower()
    for banned in ["volume", "traffic", "monthly_searches", "ranking_position"]:
        assert banned not in schema_text


def test_llm_output_rejects_extra_fields():
    data = llm_fixture()
    data["brand"]["search_volume"] = 5000
    with pytest.raises(ValidationError):
        LLMAnalysis.model_validate(data)


def _competitor(**kw):
    base = dict(name="A", website=None, description="d", target_market="t", strengths=[], weaknesses=[])
    return Competitor(**{**base, **kw})


def test_competitor_basis_default():
    assert _competitor().basis == "ai_suggestion"


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("Rival.com", "https://rival.com"),
        ("https://www.rival.co.uk/", "https://www.rival.co.uk/"),
        ("http://rival.com/about", "http://rival.com/about"),
        ("not a site", None),
        ("javascript:alert(1)", None),
        ("", None),
        (None, None),
    ],
)
def test_competitor_website_sanitized(raw, clean):
    assert _competitor(website=raw).website == clean


def test_invalid_platform_rejected():
    with pytest.raises(ValidationError):
        SearchQuery(query="q", platforms=["TikTok"], intent="i", business_relevance="b", target_audience="t")


# ---------- strict JSON schema for providers ----------

def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_strict_schema_is_provider_friendly():
    schema = strict_json_schema(LLMAnalysis)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text          # fully inlined
    assert '"default"' not in text and '"const"' not in text
    for node in _walk(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


# ---------- pricing ----------

def test_estimate_cost(monkeypatch):
    monkeypatch.setitem(PRICES_PER_MTOK, "m", Price(input=1.0, cached_input=0.1, output=4.0))
    assert estimate_cost("m", 1_000_000, 0, 500_000) == 3.0
    assert estimate_cost("m", 1_000_000, 1_000_000, 0) == 0.1
    assert estimate_cost("unknown-model", 1, 0, 1) is None


def test_default_models_have_prices():
    for provider in ["gemini", "anthropic", "openai"]:
        model = Settings(_env_file=None, llm_provider=provider).resolved_model
        assert model in PRICES_PER_MTOK


# ---------- client factory ----------

def test_missing_key_raises():
    with pytest.raises(LLMUnavailableError, match="not configured"):
        get_llm_client(Settings(_env_file=None, llm_provider="gemini", gemini_api_key=None))


@pytest.mark.parametrize("provider,key_field", [
    ("gemini", "gemini_api_key"), ("anthropic", "anthropic_api_key"), ("openai", "openai_api_key")])
def test_factory_builds_each_provider(provider, key_field):
    client = get_llm_client(Settings(_env_file=None, llm_provider=provider, **{key_field: "test-key"}))
    assert client.provider == provider
    assert client.model


# ---------- response mapping (no network) ----------

def test_anthropic_mapping_ok():
    from app.services.llm.anthropic import to_result

    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text='{"a": 1}')],
        usage=SimpleNamespace(input_tokens=100, cache_read_input_tokens=400, cache_creation_input_tokens=0, output_tokens=50),
    )
    r = to_result(resp, latency_ms=12)
    assert r.data == {"a": 1} and r.input_tokens == 500 and r.cached_input_tokens == 400
    assert r.output_tokens == 50 and r.truncated is False


def test_anthropic_mapping_truncated_and_refusal():
    from app.services.llm.anthropic import to_result

    usage = SimpleNamespace(input_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=1)
    cut = to_result(SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(type="text", text='{"a":')], usage=usage), 1)
    assert cut.truncated is True and cut.data is None
    with pytest.raises(LLMRefusedError):
        to_result(SimpleNamespace(stop_reason="refusal", content=[], usage=usage), 1)


def test_gemini_mapping():
    from app.services.llm.gemini import to_result

    usage = SimpleNamespace(prompt_token_count=300, cached_content_token_count=200, candidates_token_count=40, thoughts_token_count=10)
    ok = SimpleNamespace(text='{"b": 2}', usage_metadata=usage, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))], prompt_feedback=None)
    r = to_result(ok, 5)
    assert r.data == {"b": 2} and r.input_tokens == 300 and r.cached_input_tokens == 200 and r.output_tokens == 50
    blocked = SimpleNamespace(text=None, usage_metadata=usage, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="SAFETY"))], prompt_feedback=None)
    with pytest.raises(LLMRefusedError):
        to_result(blocked, 5)


def test_openai_mapping():
    from app.services.llm.openai import to_result

    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=60, prompt_tokens_details=SimpleNamespace(cached_tokens=256))
    msg = SimpleNamespace(content='{"c": 3}', refusal=None)
    r = to_result(SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage), 7)
    assert r.data == {"c": 3} and r.cached_input_tokens == 256 and r.output_tokens == 60
    refused = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="no"), finish_reason="stop")], usage=usage)
    with pytest.raises(LLMRefusedError):
        to_result(refused, 7)
    cut = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"c"', refusal=None), finish_reason="length")], usage=usage)
    assert to_result(cut, 7).truncated is True
