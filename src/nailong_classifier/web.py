from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import uvicorn
from PIL import Image, ImageOps
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

LOCAL_TIMEZONE: Final = ZoneInfo("Asia/Shanghai")
PUBLIC_LABELS: Final = ("nailong", "naiwa")
PUBLIC_LABEL_SET: Final = set(PUBLIC_LABELS)
LABEL_NAMES: Final = {"nailong": "奶龙", "naiwa": "奶蛙"}
SORT_ORDERS: Final = {
    "newest": "first_seen_at DESC, id DESC",
    "oldest": "first_seen_at ASC, id ASC",
    "popular": "seen_count DESC, first_seen_at DESC, id DESC",
    "rare": "seen_count ASC, first_seen_at DESC, id DESC",
}
THUMBNAIL_SIZE: Final = 480
THUMBNAIL_VERSION: Final = "v1"
IMMUTABLE_CACHE: Final = "public, max-age=31536000, immutable"
STATIC_CACHE: Final = "public, max-age=3600, stale-while-revalidate=86400"
IMAGE_MEDIA_TYPES: Final = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class GalleryAsset:
    id: int
    sha256: str
    label: str
    confidence: float
    file_name: str
    first_seen_at: str
    seen_count: int
    corrected: bool
    origin: str
    discoverer_name: str | None


class GalleryRepository:
    """读取机器人资产库；Web 端不修改分类或原图。"""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.database_path = self.state_dir / "bot.sqlite3"
        self.assets_dir = (self.state_dir / "assets").resolve()

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.is_file():
            raise FileNotFoundError(f"Asset database does not exist: {self.database_path}")
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def stats(self) -> dict[str, int]:
        counts = {label: 0 for label in PUBLIC_LABELS}
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT label, COUNT(*) AS count
                FROM assets
                WHERE label IN (?, ?)
                GROUP BY label
                """,
                PUBLIC_LABELS,
            ).fetchall()
        for row in rows:
            if row["label"] in counts:
                counts[row["label"]] = int(row["count"])
        counts["total"] = sum(counts.values())
        return counts

    def list_assets(
        self,
        *,
        label: str | None,
        page: int,
        page_size: int,
        sort: str,
    ) -> tuple[list[GalleryAsset], int]:
        where = "WHERE label = ?" if label else "WHERE label IN (?, ?)"
        parameters: list[object] = [label] if label else list(PUBLIC_LABELS)
        order_by = SORT_ORDERS[sort]
        offset = (page - 1) * page_size
        with self._connect() as connection:
            discoverer_column = self._discoverer_column(connection)
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM assets {where}",  # noqa: S608
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT id, sha256, label, confidence, file_name, first_seen_at,
                       seen_count, corrected, origin, {discoverer_column}
                FROM assets
                {where}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,  # noqa: S608
                [*parameters, page_size, offset],
            ).fetchall()
        return ([self._asset_from_row(row) for row in rows], total)

    def get_asset(self, asset_id: int) -> GalleryAsset | None:
        with self._connect() as connection:
            discoverer_column = self._discoverer_column(connection)
            row = connection.execute(
                f"""
                SELECT id, sha256, label, confidence, file_name, first_seen_at,
                       seen_count, corrected, origin, {discoverer_column}
                FROM assets
                WHERE id = ? AND label IN (?, ?)
                """,  # noqa: S608
                (asset_id, *PUBLIC_LABELS),
            ).fetchone()
        return self._asset_from_row(row) if row else None

    def asset_path(self, asset: GalleryAsset) -> Path:
        path = (self.assets_dir / asset.file_name).resolve()
        if not path.is_relative_to(self.assets_dir):
            raise FileNotFoundError("Asset path escapes the asset directory")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _asset_from_row(row: sqlite3.Row) -> GalleryAsset:
        return GalleryAsset(
            id=int(row["id"]),
            sha256=row["sha256"],
            label=row["label"],
            confidence=float(row["confidence"]),
            file_name=row["file_name"],
            first_seen_at=row["first_seen_at"],
            seen_count=int(row["seen_count"]),
            corrected=bool(row["corrected"]),
            origin=row["origin"],
            discoverer_name=row["discoverer_name"],
        )

    @staticmethod
    def _discoverer_column(connection: sqlite3.Connection) -> str:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(assets)").fetchall()
        }
        return "discoverer_name" if "discoverer_name" in columns else "NULL AS discoverer_name"


class ThumbnailCache:
    """把原图转成浏览用 WebP，并把结果永久缓存在磁盘。"""

    def __init__(self, cache_dir: Path, *, size: int = THUMBNAIL_SIZE) -> None:
        self.cache_dir = cache_dir.resolve()
        self.size = size
        # Pillow 解码时会把整张原图展开到内存。串行生成可避免多人同时翻页时
        # 几张高分辨率 PNG 叠加，把轻量 Web 服务顶到内存上限。
        self._render_lock = threading.Lock()

    def etag(self, asset: GalleryAsset) -> str:
        return f'"thumb-{THUMBNAIL_VERSION}-{self.size}-{asset.sha256}"'

    def get_or_create(self, asset: GalleryAsset, source: Path) -> Path:
        target = self._cache_path(asset)
        if target.is_file():
            return target
        with self._render_lock:
            if target.is_file():
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            # 固定临时文件名：即使进程被强制终止，下次生成也会覆盖并最终替换它，
            # 不会在缓存目录里留下按 PID 累积的残片。
            temporary = target.with_name(f".{target.name}.tmp")
            try:
                with Image.open(source) as raw_image:
                    raw_image.seek(0)
                    # JPEG/MPO 支持在解码阶段直接降采样，避免为缩略图展开完整像素。
                    raw_image.draft("RGB", (self.size, self.size))
                    orientation = raw_image.getexif().get(274)
                    # exif_transpose 在无需旋转时也会复制整张图。大 PNG 可因此瞬间
                    # 多占上百 MB；只有确实带旋转信息时才创建副本。
                    image = (
                        ImageOps.exif_transpose(raw_image)
                        if orientation not in {None, 1}
                        else raw_image
                    )
                    if image.mode not in {"RGB", "RGBA"}:
                        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
                    image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
                    image.save(temporary, "WEBP", quality=72, method=6)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    def _cache_path(self, asset: GalleryAsset) -> Path:
        name = f"{asset.sha256}-{self.size}-{THUMBNAIL_VERSION}.webp"
        return self.cache_dir / asset.sha256[:2] / name


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


def _static_dir() -> Path:
    return Path(__file__).with_name("web_assets")


def _json_etag(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f'"{hashlib.sha256(encoded).hexdigest()}"'


def _not_modified(request: Request, etag: str) -> bool:
    return request.headers.get("if-none-match") == etag


def _image_media_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    try:
        with Image.open(path) as image:
            return IMAGE_MEDIA_TYPES.get(image.format or "", "application/octet-stream")
    except (OSError, ValueError):
        return "application/octet-stream"


def _asset_json(asset: GalleryAsset) -> dict[str, object]:
    payload = asdict(asset)
    payload.pop("file_name")
    payload.pop("sha256")
    payload["label_name"] = LABEL_NAMES[asset.label]
    revision = asset.sha256[:12]
    payload["thumbnail_url"] = f"media/{asset.id}/thumbnail?rev={revision}"
    payload["original_url"] = f"media/{asset.id}/original?rev={revision}"
    return payload


def create_app(state_dir: Path) -> Starlette:
    repository = GalleryRepository(state_dir)
    thumbnails = ThumbnailCache(state_dir / "preview-cache" / "thumbnails")
    static_dir = _static_dir()

    async def index(_request: Request) -> Response:
        return HTMLResponse(
            (static_dir / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache"},
        )

    async def static_file(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in {"app.css", "app.js"}:
            raise HTTPException(status_code=404)
        return FileResponse(static_dir / name, headers={"Cache-Control": STATIC_CACHE})

    async def api_assets(request: Request) -> Response:
        query = request.query_params
        label = query.get("label") or None
        if label not in PUBLIC_LABEL_SET | {None}:
            raise HTTPException(status_code=400, detail="Unsupported label")
        sort = query.get("sort", "newest")
        if sort not in SORT_ORDERS:
            raise HTTPException(status_code=400, detail="Unsupported sort order")
        try:
            page = max(1, int(query.get("page", "1")))
            page_size = min(60, max(12, int(query.get("page_size", "36"))))
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid pagination") from error
        assets, total = await run_in_threadpool(
            repository.list_assets,
            label=label,
            page=page,
            page_size=page_size,
            sort=sort,
        )
        payload = {
            "items": [_asset_json(asset) for asset in assets],
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": page * page_size < total,
        }
        etag = _json_etag(payload)
        headers = {
            "Cache-Control": "private, no-cache",
            "ETag": etag,
        }
        if _not_modified(request, etag):
            return Response(status_code=304, headers=headers)
        return JSONResponse(payload, headers=headers)

    async def api_stats(request: Request) -> Response:
        stats = await run_in_threadpool(repository.stats)
        payload = {
            **stats,
            "generated_at": datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"),
        }
        # generated_at 不参与 ETag，让数据库没变化时客户端仍可复用响应。
        etag = _json_etag(stats)
        headers = {
            "Cache-Control": "private, no-cache",
            "ETag": etag,
        }
        if _not_modified(request, etag):
            return Response(status_code=304, headers=headers)
        return JSONResponse(payload, headers=headers)

    async def thumbnail(request: Request) -> Response:
        asset_id = request.path_params["asset_id"]
        asset = await run_in_threadpool(repository.get_asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404)
        etag = thumbnails.etag(asset)
        headers = {"Cache-Control": IMMUTABLE_CACHE, "ETag": etag}
        if _not_modified(request, etag):
            return Response(status_code=304, headers=headers)
        try:
            source = repository.asset_path(asset)
            cached = await run_in_threadpool(thumbnails.get_or_create, asset, source)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Image is unavailable") from error
        return FileResponse(cached, media_type="image/webp", headers=headers)

    async def original(request: Request) -> Response:
        asset_id = request.path_params["asset_id"]
        asset = await run_in_threadpool(repository.get_asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404)
        etag = f'"original-{asset.sha256}"'
        headers = {"Cache-Control": IMMUTABLE_CACHE, "ETag": etag}
        if _not_modified(request, etag):
            return Response(status_code=304, headers=headers)
        try:
            source = repository.asset_path(asset)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Image is unavailable") from error
        media_type = await run_in_threadpool(_image_media_type, source)
        return FileResponse(source, media_type=media_type, headers=headers)

    async def health(_request: Request) -> Response:
        try:
            stats = await run_in_threadpool(repository.stats)
        except (FileNotFoundError, sqlite3.Error) as error:
            return JSONResponse({"status": "error", "detail": str(error)}, status_code=503)
        return JSONResponse({"status": "ok", "assets": stats["total"]})

    app = Starlette(
        debug=False,
        middleware=[Middleware(SecurityHeadersMiddleware)],
        routes=[
            Route("/", index),
            Route("/static/{name:str}", static_file),
            Route("/api/assets", api_assets),
            Route("/api/stats", api_stats),
            Route("/media/{asset_id:int}/thumbnail", thumbnail),
            Route("/media/{asset_id:int}/original", original),
            Route("/healthz", health),
        ],
    )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nailong image gallery")
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "8765")))
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.getenv("BOT_STATE_DIR", "runtime")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        create_app(args.state_dir),
        host=args.host,
        port=args.port,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
