import time

from starlette.middleware.base import BaseHTTPMiddleware


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Process-Time-ms"] = str(duration_ms)

        return response