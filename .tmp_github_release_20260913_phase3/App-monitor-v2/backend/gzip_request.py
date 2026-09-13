"""gzip圧縮された端末JSONをサイズ制限付きで展開するASGI middleware。"""

from __future__ import annotations

import gzip

from fastapi.responses import JSONResponse

class GzipRequestMiddleware:
    def __init__(self, app, *, max_compressed_bytes: int = 2_000_000, max_body_bytes: int = 16_000_000):
        self.app = app
        self.max_compressed_bytes = max_compressed_bytes
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if headers.get(b"content-encoding", b"").lower() != b"gzip":
            await self.app(scope, receive, send)
            return

        compressed = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            compressed.extend(message.get("body", b""))
            if len(compressed) > self.max_compressed_bytes:
                await self._error(scope, receive, send, 413, "compressed_request_too_large")
                return
            more_body = message.get("more_body", False)
        try:
            body = gzip.decompress(bytes(compressed))
        except (gzip.BadGzipFile, EOFError, OSError):
            await self._error(scope, receive, send, 400, "invalid_gzip_body")
            return
        if len(body) > self.max_body_bytes:
            await self._error(scope, receive, send, 413, "decompressed_request_too_large")
            return

        emitted = False

        async def decoded_receive():
            nonlocal emitted
            if emitted:
                return {"type": "http.request", "body": b"", "more_body": False}
            emitted = True
            return {"type": "http.request", "body": body, "more_body": False}

        rewritten_headers = [
            (key, value)
            for key, value in scope.get("headers", [])
            if key.lower() not in {b"content-encoding", b"content-length"}
        ]
        rewritten_headers.append((b"content-length", str(len(body)).encode("ascii")))
        decoded_scope = {**scope, "headers": rewritten_headers}
        await self.app(decoded_scope, decoded_receive, send)

    @staticmethod
    async def _error(scope, receive, send, status_code: int, code: str):
        response = JSONResponse(
            status_code=status_code,
            content={
                "message_type": "error",
                "schema_version": "1.0",
                "error": {"code": code, "message": "gzip request body could not be accepted"},
            },
        )
        await response(scope, receive, send)
