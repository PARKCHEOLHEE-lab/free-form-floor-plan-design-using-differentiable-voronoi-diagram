#!/usr/bin/env python3
"""Static server that sets COOP/COEP so the page is cross-origin isolated,
which SharedArrayBuffer (and thus the wasm-bindgen-rayon thread pool) requires.
A plain `python -m http.server` does NOT send these, so the threaded wasm fails
to instantiate. Run from the web/ directory: `python3 serve.py [port]`."""
import http.server
import os
import socketserver
import sys

# Always serve the directory this script lives in (web/), regardless of CWD.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"serving web/ on http://localhost:{PORT} with COOP/COEP (cross-origin isolated)")
    httpd.serve_forever()
