from __future__ import annotations

import http.client
import importlib.util
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@contextmanager
def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_proxy_forwards_only_bounded_sticker_requests_without_caching(tmp_path, monkeypatch):
    monkeypatch.setenv("NAILONG_PROXY_TOKEN", "local-test-token")
    path = Path(__file__).parents[1] / "deploy" / "nailong-devbox-proxy.py"
    spec = importlib.util.spec_from_file_location("gallery_proxy_test", path)
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    proxy.UPSTREAM_HOST = "127.0.0.1"
    proxy.CACHE_DIR = tmp_path
    observed = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            observed.append((self.path, self.headers["X-Nailong-Proxy-Token"], body))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "7")
            self.end_headers()
            self.wfile.write(b"pngdata")

        def log_message(self, *_args):
            pass

    def post(port, path, body, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request("POST", path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.getheader("Cache-Control"), response.read()
        finally:
            connection.close()

    with serve(Upstream) as upstream_port:
        proxy.UPSTREAM_PORT = upstream_port
        with serve(proxy.GalleryProxyHandler) as port:
            payload = '{"text":"奶龙批准"}'.encode()
            for _ in range(2):
                status, cache, body = post(
                    port, "/api/milk-text", payload, {"Content-Type": "application/json"}
                )
                assert (status, cache, body) == (200, "no-store", b"pngdata")
            assert post(port, "/api/milk-text", b"x" * 4097)[0] == 413
            assert post(port, "/api/milk-text", b"", {"Content-Length": "-1"})[0] == 413
            assert post(port, "/api/milk-text", b"", {"Content-Length": "bad"})[0] == 411
            assert post(port, "/api/stats", b"{}")[0] == 405
    assert observed == [("/nailong/api/milk-text", "local-test-token", payload)] * 2
    assert not list(tmp_path.iterdir())
