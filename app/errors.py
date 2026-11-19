"""Application errors.

Every expected failure is an `AppError` subclass carrying its HTTP status and a
stable machine-readable code. One handler in `main.py` turns any of them into
`{"success": false, "error": {"code": ..., "message": ...}}`.
"""


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidURLError(AppError):
    status_code = 422
    code = "INVALID_URL"


class BlockedAddressError(AppError):
    status_code = 400
    code = "BLOCKED_ADDRESS"


class SiteUnreachableError(AppError):
    status_code = 502
    code = "SITE_UNREACHABLE"


class SiteTimeoutError(AppError):
    status_code = 504
    code = "SITE_TIMEOUT"


class UnsupportedContentError(AppError):
    status_code = 422
    code = "UNSUPPORTED_CONTENT"


class InsufficientContentError(AppError):
    status_code = 422
    code = "INSUFFICIENT_CONTENT"


class LLMUnavailableError(AppError):
    status_code = 502
    code = "LLM_UNAVAILABLE"


class LLMOutputError(AppError):
    status_code = 502
    code = "LLM_INVALID_OUTPUT"


class LLMRefusedError(AppError):
    status_code = 422
    code = "LLM_REFUSED"


class BudgetExceededError(AppError):
    status_code = 503
    code = "BUDGET_EXCEEDED"
