import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.errors import AppError
from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.llm.base import get_llm_client
from app.services.website_extractor import build_http_client

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

    @target.exception_handler(RateLimitExceeded)
    async def handle_rate_limit(_: Request, exc: RateLimitExceeded) -> JSONResponse:
        return _error(429, "RATE_LIMITED", f"Too many requests ({exc.detail}). Please wait and try again.")

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


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared HTTP client (connection pooling) for the app's lifetime.
    app.state.http = build_http_client(get_settings())
    yield
    await app.state.http.aclose()
    app.state.http = None
    app.state.service = None


app = FastAPI(
    title="AI Link Analysis API",
    version="1.0.0",
    description=(
        "Analyzes a website URL with an LLM and returns competitor, search-query, "
        "SEO, services and target-audience insights as structured JSON.\n\n"
        "Every response includes a `meta` block with token usage, estimated cost and latency."
    ),
    lifespan=lifespan,
)
app.state.limiter = limiter
register_exception_handlers(app)


def get_analysis_service(request: Request) -> AnalysisService:
    """One service per app. The LLM client inside it is created lazily (first LLM call),
    so the app, /docs and every pre-LLM check work even when no API key is configured."""
    state = request.app.state
    if getattr(state, "service", None) is None:
        settings = get_settings()
        http = getattr(state, "http", None) or build_http_client(settings)
        state.service = AnalysisService(settings, lambda: get_llm_client(settings), http)
    return state.service


_ERRORS = {
    400: "URL points to a private or internal address (BLOCKED_ADDRESS).",
    422: "Invalid request or URL, unsupported content, too little text, or the AI declined "
         "(INVALID_REQUEST, INVALID_URL, UNSUPPORTED_CONTENT, INSUFFICIENT_CONTENT, LLM_REFUSED).",
    429: "Rate limit exceeded (RATE_LIMITED).",
    502: "Website unreachable or AI provider failed (SITE_UNREACHABLE, LLM_UNAVAILABLE, LLM_INVALID_OUTPUT).",
    503: "Daily AI budget reached (BUDGET_EXCEEDED).",
    504: "Website took too long to respond (SITE_TIMEOUT).",
}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    responses={status: {"model": ErrorResponse, "description": text} for status, text in _ERRORS.items()},
    summary="Analyze a website URL",
    tags=["analysis"],
)
@limiter.limit(lambda: get_settings().rate_limit)
async def link_analysis(
    request: Request,  # required by the rate limiter
    body: LinkAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> LinkAnalysisResponse:
    """Fetch the website, extract its content and return a five-section AI analysis."""
    return await service.analyze(body.url)


_DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def demo_page() -> FileResponse:
    """A small UI for non-technical reviewers; the API itself is POST /link-analysis."""
    return FileResponse(_DEMO_PAGE, media_type="text/html")


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
