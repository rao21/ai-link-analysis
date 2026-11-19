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
