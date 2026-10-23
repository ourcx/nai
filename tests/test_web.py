from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from starlette.testclient import TestClient

from nailong_classifier.storage import BotStore
from nailong_classifier.web import GalleryRepository, ThumbnailCache, _image_media_type, create_app


def make_image_bytes(color: str, *, size: tuple[int, int] = (900, 600)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def record_asset(
    store: BotStore,
    *,
    message_id: str,
    color: str,
    label: str,
    discoverer_name: str | None = None,
):
    return store.record_observation(
        image_bytes=make_image_bytes(color),
        source_message_id=message_id,
        chat_id="chat-web",
        chat_type="group",
        sender_id=f"sender-{message_id}",
        predicted_label=label,
        confidence=0.91,
        accepted=True,
        discoverer_name=discoverer_name,
    ).asset


def test_gallery_repository_filters_sorts_and_counts(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    first = record_asset(store, message_id="one", color="yellow", label="nailong")
    record_asset(store, message_id="two", color="pink", label="naiwa")
    popular = record_asset(store, message_id="three", color="blue", label="other")
    for index in range(3):
        store.record_observation(
            image_bytes=popular.path.read_bytes(),
            source_message_id=f"repeat-{index}",
            chat_id="chat-web",
            chat_type="group",
            sender_id="repeat-sender",
            predicted_label="other",
            confidence=0.88,
            accepted=True,
        )

    repository = GalleryRepository(tmp_path)

    assert repository.stats() == {"nailong": 1, "naiwa": 1, "total": 2}
    nailong, total = repository.list_assets(
        label="nailong", page=1, page_size=12, sort="newest"
    )
    assert total == 1
    assert [asset.id for asset in nailong] == [first.id]
    assets, total = repository.list_assets(
        label=None, page=1, page_size=12, sort="popular"
    )
    assert total == 2
    assert {asset.label for asset in assets} == {"nailong", "naiwa"}
    assert all(asset.id != popular.id for asset in assets)


def test_thumbnail_cache_generates_small_webp_once(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    original = record_asset(
        store,
        message_id="thumb",
        color="yellow",
        label="nailong",
    )
    repository = GalleryRepository(tmp_path)
    asset = repository.get_asset(original.id)
    cache = ThumbnailCache(tmp_path / "preview-cache", size=320)

    cached = cache.get_or_create(asset, repository.asset_path(asset))
    first_mtime = cached.stat().st_mtime_ns
    same_cached = cache.get_or_create(asset, repository.asset_path(asset))

    assert cached == same_cached
    assert same_cached.stat().st_mtime_ns == first_mtime
    assert cached.stat().st_size < original.path.stat().st_size
    with Image.open(cached) as image:
        assert image.format == "WEBP"
        assert max(image.size) == 320


def test_thumbnail_cache_serializes_different_image_decodes(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    first = record_asset(store, message_id="first", color="yellow", label="nailong")
    second = record_asset(store, message_id="second", color="pink", label="naiwa")
    repository = GalleryRepository(tmp_path)
    assets = [repository.get_asset(first.id), repository.get_asset(second.id)]
    cache = ThumbnailCache(tmp_path / "preview-cache", size=320)
    original_open = Image.open
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class TrackedImage:
        def __init__(self, image):
            self.image = image

        def __enter__(self):
            nonlocal active, max_active
            entered = self.image.__enter__()
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            return entered

        def __exit__(self, *args):
            nonlocal active
            try:
                return self.image.__exit__(*args)
            finally:
                with state_lock:
                    active -= 1

    def tracked_open(*args, **kwargs):
        return TrackedImage(original_open(*args, **kwargs))

    with (
        patch("nailong_classifier.web.Image.open", side_effect=tracked_open),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        results = list(
            executor.map(
                lambda asset: cache.get_or_create(asset, repository.asset_path(asset)),
                assets,
            )
        )

    assert all(result.is_file() for result in results)
    assert max_active == 1


def test_gallery_api_and_media_cache_headers(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    asset = record_asset(
        store,
        message_id="api",
        color="pink",
        label="naiwa",
        discoverer_name="首发者",
    )
    client = TestClient(create_app(tmp_path))

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    assert stats.json()["total"] == 1
    stats_cached = client.get("/api/stats", headers={"If-None-Match": stats.headers["etag"]})
    assert stats_cached.status_code == 304

    response = client.get("/api/assets?label=naiwa&page=1&page_size=12")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["has_more"] is False
    assert payload["items"][0]["discoverer_name"] == "首发者"
    assert "file_name" not in payload["items"][0]
    assert "sha256" not in payload["items"][0]
    assert payload["items"][0]["thumbnail_url"].startswith("media/")
    assert payload["items"][0]["original_url"].startswith("media/")

    thumbnail = client.get(payload["items"][0]["thumbnail_url"])
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/webp"
    assert "immutable" in thumbnail.headers["cache-control"]
    thumbnail_cached = client.get(
        payload["items"][0]["thumbnail_url"],
        headers={"If-None-Match": thumbnail.headers["etag"]},
    )
    assert thumbnail_cached.status_code == 304

    original = client.get(payload["items"][0]["original_url"])
    assert original.status_code == 200
    assert original.content == asset.path.read_bytes()
    assert "immutable" in original.headers["cache-control"]


def test_gallery_page_is_lazy_and_input_is_validated(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    record_asset(store, message_id="one", color="yellow", label="nailong")
    client = TestClient(create_app(tmp_path))

    page = client.get("/")
    assert page.status_code == 200
    assert "奶龙奶蛙图鉴" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert "loading=\"lazy\"" not in page.text  # 图片卡由 API 返回后再创建。
    assert "http://" not in page.text
    assert "https://" not in page.text
    assert client.get("/api/assets?label=unknown").status_code == 400
    assert client.get("/api/assets?label=other").status_code == 400
    assert client.get("/api/assets?page=bad").status_code == 400
    assert client.get("/static/not-found.js").status_code == 404


def test_other_assets_are_never_exposed(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    record_asset(store, message_id="milk", color="yellow", label="nailong")
    other = record_asset(store, message_id="private", color="blue", label="other")
    client = TestClient(create_app(tmp_path))

    stats = client.get("/api/stats").json()
    assert stats == {
        "nailong": 1,
        "naiwa": 0,
        "total": 1,
        "generated_at": stats["generated_at"],
    }
    payload = client.get("/api/assets?page_size=12").json()
    assert payload["total"] == 1
    assert all(item["label"] != "other" for item in payload["items"])
    assert client.get(f"/media/{other.id}/thumbnail").status_code == 404
    assert client.get(f"/media/{other.id}/original").status_code == 404


def test_gallery_health_reports_missing_database(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "error"


def test_unknown_file_suffix_uses_decoded_image_media_type(tmp_path: Path) -> None:
    image_path = tmp_path / "camera-upload.bin"
    image_path.write_bytes(make_image_bytes("yellow"))

    assert _image_media_type(image_path) == "image/png"
