import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog


logger = structlog.get_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request has a request_id (header: X-Request-ID).
    Stores it in structlog's context so all logs correlate.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response
        try:
            response = await call_next(request)
        finally:
            # Clear per-request contextvars so they don't leak across requests
            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Minimal access log with latency in ms, method, path, status, client ip.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "http.access",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            status_code=response.status_code,
            client_ip=request.client.host if request.client else None,
            latency_ms=round(latency_ms, 2),
        )
        return response
