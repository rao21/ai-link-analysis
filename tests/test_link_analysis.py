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
