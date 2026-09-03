from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Callable


def make_handler(state, html: str) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_root_headers(self, content_length: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(content_length))
            self.end_headers()

        def do_HEAD(self):
            if self.path == "/":
                self.send_root_headers(len(html.encode("utf-8")))
                return
            self.send_error(404)

        def do_GET(self):
            if self.path == "/":
                body = html.encode("utf-8")
                self.send_root_headers(len(body))
                self.wfile.write(body)
                return
            if self.path == "/api/snapshot":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def log_message(self, _format, *_args):
            return

    return Handler
