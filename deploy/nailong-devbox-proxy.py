from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

LISTEN_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("PROXY_PORT", "8765"))
UPSTREAM_HOST = os.getenv("UPSTREAM_HOST", "1.14.151.75")
UPSTREAM_PORT = int(os.getenv("UPSTREAM_PORT", "80"))
UPSTREAM_PREFIX = os.getenv("UPSTREAM_PREFIX", "/nailong").rstrip("/")
ORIGIN_TOKEN = os.environ["NAILONG_PROXY_TOKEN"]
CACHE_DIR = Path(os.getenv("PROXY_CACHE_DIR", "/var/cache/nailong-gallery-proxy"))
CACHEABLE_MEDIA_PATTERN = re.compile(r"^/media/\d+/thumbnail$")
CACHE_LOCKS_GUARD = threading.Lock()
CACHE_LOCKS: dict[str, threading.Lock] = {}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
FORWARDED_REQUEST_HEADERS = {
    "accept",
    "if-modified-since",
    "if-none-match",
    "range",
    "user-agent",
}


def cache_lock(cache_key: str) -> threading.Lock:
    with CACHE_LOCKS_GUARD:
        return CACHE_LOCKS.setdefault(cache_key, threading.Lock())


class GalleryProxyServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class GalleryProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NailongGalleryProxy"
    sys_version = ""

    def do_GET(self) -> None:
        self._proxy(send_body=True)

    def do_HEAD(self) -> None:
        self._proxy(send_body=False)

    def do_POST(self) -> None:
        self.send_error(405, "Method Not Allowed")

    def _proxy(self, *, send_body: bool) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/_proxy_health":
            self._health()
            return

        cache_paths = self._cache_paths(parsed.path, parsed.query)
        lock = cache_lock(cache_paths[0]) if cache_paths else None
        if lock:
            lock.acquire()
            try:
                cache_hit = self._serve_cached(*cache_paths, send_body=send_body)
            except BrokenPipeError:
                lock.release()
                return
            if cache_hit:
                lock.release()
                return

        upstream_path = f"{UPSTREAM_PREFIX}/{parsed.path.lstrip('/')}"
        if parsed.query:
            upstream_path = f"{upstream_path}?{parsed.query}"
        headers = {
            "Host": UPSTREAM_HOST,
            "X-Nailong-Proxy-Token": ORIGIN_TOKEN,
            "X-Forwarded-For": self.client_address[0],
            "X-Forwarded-Proto": "http",
        }
        for name in FORWARDED_REQUEST_HEADERS:
            value = self.headers.get(name)
            if value:
                headers[name] = value

        connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=30)
        response_started = False
        temporary_body: Path | None = None
        try:
            connection.request(self.command, upstream_path, headers=headers)
            response = connection.getresponse()
            response_headers = [
                (name, value)
                for name, value in response.getheaders()
                if name.lower() not in HOP_BY_HOP_HEADERS | {"date", "server"}
            ]
            self.send_response(response.status, response.reason)
            for name, value in response_headers:
                self.send_header(name, value)
            if cache_paths:
                self.send_header("X-Proxy-Cache", "MISS")
            self.end_headers()
            response_started = True
            should_cache = bool(cache_paths and self.command == "GET" and response.status == 200)
            if should_cache:
                _, body_path, metadata_path = cache_paths
                body_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_body = body_path.with_name(
                    f".{body_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                with temporary_body.open("wb") as cache_file:
                    while chunk := response.read(64 * 1024):
                        cache_file.write(chunk)
                        if send_body:
                            self.wfile.write(chunk)
                temporary_body.replace(body_path)
                metadata = {
                    "status": response.status,
                    "reason": response.reason,
                    "headers": response_headers,
                    "stored_at": int(time.time()),
                }
                temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
                temporary_metadata.write_text(json.dumps(metadata), encoding="utf-8")
                temporary_metadata.replace(metadata_path)
            elif send_body:
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionError, TimeoutError):
            if not response_started and not self.wfile.closed:
                self.send_error(502, "Upstream unavailable")
        finally:
            if temporary_body:
                temporary_body.unlink(missing_ok=True)
            connection.close()
            if lock:
                lock.release()

    def _cache_paths(self, path: str, query: str) -> tuple[str, Path, Path] | None:
        if self.command not in {"GET", "HEAD"} or self.headers.get("Range"):
            return None
        if not (path.startswith("/static/") or CACHEABLE_MEDIA_PATTERN.fullmatch(path)):
            return None
        cache_key = hashlib.sha256(f"{path}?{query}".encode()).hexdigest()
        cache_root = CACHE_DIR / cache_key[:2]
        return cache_key, cache_root / f"{cache_key}.body", cache_root / f"{cache_key}.json"

    def _serve_cached(
        self,
        _cache_key: str,
        body_path: Path,
        metadata_path: Path,
        *,
        send_body: bool,
    ) -> bool:
        if not body_path.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            response_headers = metadata["headers"]
        except (OSError, KeyError, TypeError, ValueError):
            body_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return False

        etag = next(
            (value for name, value in response_headers if name.lower() == "etag"),
            None,
        )
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            for name, value in response_headers:
                if name.lower() in {"cache-control", "etag", "expires", "last-modified", "vary"}:
                    self.send_header(name, value)
            self.send_header("X-Proxy-Cache", "HIT")
            self.end_headers()
            return True

        self.send_response(int(metadata["status"]), metadata.get("reason"))
        for name, value in response_headers:
            self.send_header(name, value)
        self.send_header("X-Proxy-Cache", "HIT")
        self.end_headers()
        if send_body:
            with body_path.open("rb") as cache_file:
                while chunk := cache_file.read(64 * 1024):
                    self.wfile.write(chunk)
        return True

    def _health(self) -> None:
        payload = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def log_message(self, message_format: str, *args: object) -> None:
        print(f"{self.address_string()} - {message_format % args}", flush=True)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    server = GalleryProxyServer((LISTEN_HOST, LISTEN_PORT), GalleryProxyHandler)
    print(f"proxy listening on {LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
