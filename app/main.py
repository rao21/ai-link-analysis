import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import AppError

logger = logging.getLogger("link_analysis")

_HTTP_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": {"code": code, "message": message}},
    )


def register_exception_handlers(target: FastAPI) -> None:
    """Make every error response share one JSON shape."""

    @target.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message)

    @target.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        return _error(exc.status_code, code, str(exc.detail))

    @target.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        message = f"{field}: {first.get('msg', 'invalid request')}" if field else "Invalid request body"
        return _error(422, "INVALID_REQUEST", message)

    @target.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Full traceback goes to the server log only; the client gets a generic message.
        logger.exception("Unhandled error", exc_info=exc)
        return _error(500, "INTERNAL_ERROR", "An unexpected error occurred. Please try again later.")


app = FastAPI(
    title="AI Link Analysis API",
    version="1.0.0",
    description=(
        "Analyzes a website URL with an LLM and returns competitor, search-query, "
        "SEO, services and target-audience insights as structured JSON."
    ),
)
register_exception_handlers(app)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
