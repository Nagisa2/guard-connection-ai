from __future__ import annotations

import gzip
import json
import unittest

from gzip_request import GzipRequestMiddleware

async def _call_middleware(body: bytes, *, encoding: bytes = b"gzip"):
    captured = {}

    async def app(scope, receive, _send):
        captured["scope"] = scope
        captured["message"] = await receive()

    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        return next(messages)

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/device",
        "headers": [(b"content-encoding", encoding), (b"content-length", str(len(body)).encode())],
    }
    await GzipRequestMiddleware(app)(scope, receive, send)
    return captured, sent

class GzipRequestMiddlewareTest(unittest.IsolatedAsyncioTestCase):
    async def test_gzip_json_is_decoded_and_headers_are_rewritten(self) -> None:
        original = json.dumps({"schema_version": "1.0", "ppg0": [1, 2]}).encode()
        captured, sent = await _call_middleware(gzip.compress(original))
        self.assertFalse(sent)
        self.assertEqual(captured["message"]["body"], original)
        headers = dict(captured["scope"]["headers"])
        self.assertNotIn(b"content-encoding", headers)
        self.assertEqual(int(headers[b"content-length"]), len(original))

    async def test_invalid_gzip_returns_structured_400(self) -> None:
        captured, sent = await _call_middleware(b"not-gzip")
        self.assertFalse(captured)
        self.assertEqual(sent[0]["status"], 400)

if __name__ == "__main__":
    unittest.main()
