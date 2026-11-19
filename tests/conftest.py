import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


FAKE_DNS: dict[str, list[str]] = {
    "localhost": ["127.0.0.1"],
    "mixed.test": ["8.8.8.8", "10.0.0.1"],
    "example.com": ["93.184.216.34"],
}


@pytest.fixture
def fake_resolver(monkeypatch):
    """Replace DNS with a fixed table. Unknown `*.test` hosts resolve to a public IP."""

    async def fake_resolve(host: str) -> list[str]:
        if host in FAKE_DNS:
            return FAKE_DNS[host]
        if host.endswith(".test"):
            return ["93.184.216.34"]
        from app.errors import SiteUnreachableError

        raise SiteUnreachableError("Domain could not be resolved")

    monkeypatch.setattr("app.utils.validators.resolve_host", fake_resolve)
    return FAKE_DNS


# ---------------------------------------------------------------- LLM fakes

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.services.llm.base import LLMResult

FIXTURES = Path(__file__).parent / "fixtures"


def llm_valid() -> dict:
    return json.loads((FIXTURES / "llm_valid.json").read_text())


@dataclass
class LLMCall:
    system: str
    user: str


@dataclass
class FakeLLMClient:
    """Returns queued outputs in order (dict -> valid result, LLMResult -> as is)."""

    outputs: list = field(default_factory=list)
    calls: list[LLMCall] = field(default_factory=list)
    provider: str = "fake"
    model: str = "fake-model"

    async def generate_json(self, system, user, schema):
        self.calls.append(LLMCall(system, user))
        item = self.outputs.pop(0) if self.outputs else llm_valid()
        if isinstance(item, LLMResult):
            return item
        return LLMResult(data=copy.deepcopy(item), input_tokens=5000, cached_input_tokens=1000,
                         output_tokens=1500, latency_ms=900)


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()
