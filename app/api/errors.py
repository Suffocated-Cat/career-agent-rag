"""
Consistent error handling for the API.

Wraps the framework's default error responses in the same envelope shape as
successful responses: ``{"status": "error", "error": {type, message, detail}}``.
Covers request validation (422), explicit HTTPExceptions, and any unhandled
exception (500, without leaking internals).
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.common import ErrorDetail, ErrorResponse


def _envelope(type_: str, message: str, detail: Any | None = None) -> dict:
    """Build the error-response body."""
    return ErrorResponse(
        error=ErrorDetail(type=type_, message=message, detail=detail)
    ).model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    """Register the consistent error handlers on *app*."""

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "Request validation failed",
                jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "Internal server error"),
        )
