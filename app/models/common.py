from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Structured error information."""

    type: str  # e.g. "validation_error" | "http_error" | "internal_error"
    message: str
    detail: Any | None = None


class ErrorResponse(BaseModel):
    """Consistent error envelope, mirroring the success ``{status, data}`` shape."""

    status: str = "error"
    error: ErrorDetail
