from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.errors import BlockedAddressError
from app.main import app


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_available(client):
    assert client.get("/docs").status_code == 200


def _client_with_route(path: str, exc: Exception) -> TestClient:
    """Register a throwaway route that raises `exc`, to test the global handlers."""

    def raiser():
        raise exc

    if not any(getattr(r, "path", None) == path for r in app.router.routes):
        app.add_api_route(path, raiser, methods=["GET"], include_in_schema=False)
    return TestClient(app, raise_server_exceptions=False)


def test_app_error_shape():
    client = _client_with_route("/_test/raise", BlockedAddressError("x"))
    response = client.get("/_test/raise")
    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": {"code": "BLOCKED_ADDRESS", "message": "x"},
    }


def test_unhandled_error_hides_details():
    client = _client_with_route("/_test/boom", RuntimeError("secret"))
    response = client.get("/_test/boom")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text


def test_request_validation_error_shape():
    probe = FastAPI()  # isolated app so we don't depend on Task 7's route
    from app.main import register_exception_handlers

    register_exception_handlers(probe)

    @probe.get("/needs-int")
    def needs_int(n: int):
        return {"n": n}

    response = TestClient(probe).get("/needs-int?n=abc")
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_unknown_path_uses_error_shape(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "error": {"code": "NOT_FOUND", "message": "Not Found"},
    }


# ---------------------------------------------------------------- POST /link-analysis

import pytest

from app.config import Settings, get_settings
from app.main import get_analysis_service
from app.services.analysis_service import AnalysisService
from app.services.website_extractor import build_http_client
from tests.conftest import FIXTURES, FakeLLMClient

HTML = {"content-type": "text/html; charset=utf-8"}


@pytest.fixture
def client_with_fake(respx_mock, fake_resolver):
    respx_mock.get("https://acme.test/").respond(200, headers=HTML, text=(FIXTURES / "sample_site.html").read_text())
    respx_mock.get("https://acme.test/services").respond(200, headers=HTML, text=(FIXTURES / "services_page.html").read_text())
    respx_mock.get("https://acme.test/about").respond(404, headers=HTML, text="nope")
    settings = Settings(_env_file=None)
    service = AnalysisService(settings, FakeLLMClient(), build_http_client(settings))
    app.dependency_overrides[get_analysis_service] = lambda: service
    app.state.limiter.reset()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_link_analysis_ok(client_with_fake):
    response = client_with_fake.post("/link-analysis", json={"url": "https://acme.test"})
    body = response.json()
    assert response.status_code == 200 and body["success"] is True
    assert body["website"] == "https://acme.test/"
    assert set(body["analysis"]) == {
        "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}
    assert body["meta"]["llm_attempts"] == 1


def test_invalid_url_works_without_api_key(client):
    app.state.service = None
    response = client.post("/link-analysis", json={"url": "http://localhost/"})
    app.state.service = None
    assert response.json()["error"]["code"] == "BLOCKED_ADDRESS"  # no LLM key needed for cheap checks


def test_missing_url_field(client):
    response = client.post("/link-analysis", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_invalid_url(client_with_fake):
    response = client_with_fake.post("/link-analysis", json={"url": "ftp://x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_URL"


def test_private_url(client_with_fake):
    response = client_with_fake.post("/link-analysis", json={"url": "http://127.0.0.1/"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BLOCKED_ADDRESS"


def test_rate_limited(client_with_fake, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    try:
        codes = [client_with_fake.post("/link-analysis", json={"url": "ftp://x"}).status_code for _ in range(3)]
        last = client_with_fake.post("/link-analysis", json={"url": "ftp://x"})
    finally:
        get_settings.cache_clear()
    assert codes[:2] == [422, 422] and codes[2] == 429
    assert last.json()["error"]["code"] == "RATE_LIMITED"


def test_no_api_key_gives_clear_error(monkeypatch, fake_resolver, respx_mock):
    respx_mock.get("https://acme.test/").respond(200, headers=HTML, text=(FIXTURES / "sample_site.html").read_text())
    respx_mock.get(url__regex=r"https://acme\.test/.+").respond(404, headers=HTML, text="nope")
    app.state.service = None
    get_settings.cache_clear()
    for var in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    app.state.limiter.reset()
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            assert c.get("/docs").status_code == 200  # app boots without a key
            response = c.post("/link-analysis", json={"url": "https://acme.test"})
    finally:
        get_settings.cache_clear()
        app.state.service = None
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_UNAVAILABLE"


def test_root_serves_demo_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="analyze-form"' in response.text
    assert "/link-analysis" in response.text and "/docs" in response.text


def test_openapi_documents_error_responses(client):
    spec = client.get("/openapi.json").json()
    responses = spec["paths"]["/link-analysis"]["post"]["responses"]
    for status in ["200", "400", "422", "429", "502", "503", "504"]:
        assert status in responses


def _req(headers: dict, client_host: str = "203.0.113.9"):
    from starlette.requests import Request

    scope = {"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": (client_host, 1234), "method": "GET", "path": "/"}
    return Request(scope)


def test_rate_limit_key_ignores_spoofed_forwarded_for_by_default(monkeypatch):
    from app.main import client_ip

    get_settings.cache_clear()
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "false")
    try:
        assert client_ip(_req({"X-Forwarded-For": "1.2.3.4"})) == "203.0.113.9"
    finally:
        get_settings.cache_clear()


def test_rate_limit_key_behind_proxy_uses_rightmost_forwarded_for(monkeypatch):
    from app.main import client_ip

    get_settings.cache_clear()
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    try:
        # client spoofs "1.2.3.4"; the platform proxy appends the real address last
        assert client_ip(_req({"X-Forwarded-For": "1.2.3.4, 198.51.100.7"}, client_host="10.0.0.2")) == "198.51.100.7"
        assert client_ip(_req({}, client_host="10.0.0.2")) == "10.0.0.2"
    finally:
        get_settings.cache_clear()
