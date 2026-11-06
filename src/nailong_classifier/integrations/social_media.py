from __future__ import annotations

import html
import io
import json
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx
import numpy as np
from PIL import Image

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
PAGE_DOMAINS = ("douyin.com", "iesdouyin.com", "xiaohongshu.com", "xhslink.com")
MEDIA_DOMAINS = (
    *PAGE_DOMAINS, "douyinpic.com", "douyincdn.com", "douyinvod.com",
    "byteimg.com", "ibytedtos.com", "bytecdn.cn", "bytecdn.com",
    "pstatp.com", "snssdk.com", "amemv.com", "bytedance.net", "xhscdn.com",
)
MAX_IMAGES = 100
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 150 * 1024 * 1024
VIDEO_SAMPLES = 12


class SocialMediaError(RuntimeError):
    """可直接返回给用户的解析失败原因，不包含 Cookie 或签名 URL。"""


@dataclass(frozen=True)
class SocialImage:
    source_key: str
    kind: str
    content: bytes | None


def _host_allowed(url: str, domains: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https" and parsed.port in {None, 443}
        and not parsed.username and not parsed.password
        and any(host == domain or host.endswith(f".{domain}") for domain in domains)
    )


def find_social_link(text: str) -> str | None:
    """支持平台分享文案中的链接，只处理单条作品，不接受主页或任意下载地址。"""
    for match in re.finditer(r"https://[^\s<>\"'`，。；！？、]+", text):
        url = html.unescape(match.group()).rstrip(")]}）】")
        try:
            if _host_allowed(url, PAGE_DOMAINS):
                return url
        except ValueError:
            continue
    return None


def _load_cookies(client: httpx.Client, path: Path | None, domain: str) -> None:
    if path is None:
        return
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("cookies")
        if not isinstance(values, list):
            raise ValueError
        for item in values:
            if not isinstance(item, dict):
                continue
            cookie_domain = str(item.get("domain", "")).lstrip(".").lower()
            name, value = item.get("name"), item.get("value")
            if (
                cookie_domain == domain or cookie_domain.endswith(f".{domain}")
            ) and isinstance(name, str) and isinstance(value, str):
                if re.fullmatch(r"[A-Za-z0-9_-]+", name) and not re.search(r"[\r\n]", value):
                    client.cookies.set(name, value, domain=str(item["domain"]), path="/")
    except (OSError, ValueError, TypeError) as exc:
        raise SocialMediaError("读取平台 Cookie 文件失败，请使用浏览器导出的 Cookie JSON。") from exc


def _download(
    client: httpx.Client, url: str, *, domains: tuple[str, ...],
    max_bytes: int, path: Path | None = None,
) -> tuple[bytes, str]:
    # 每次重定向都检查目标域名；Cookie 由 cookie jar 按域名发送，不传给无关 CDN。
    for _ in range(6):
        if not _host_allowed(url, domains):
            raise SocialMediaError("平台返回了不支持的下载地址。")
        with client.stream("GET", url) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            response.raise_for_status()
            if int(response.headers.get("content-length") or 0) > max_bytes:
                raise SocialMediaError("平台媒体超过下载大小限制。")
            size = Decimal(0)
            with (path.open("wb") if path else io.BytesIO()) as output:
                for chunk in response.iter_bytes():
                    size += Decimal(len(chunk))
                    if size > Decimal(max_bytes):
                        raise SocialMediaError("平台媒体超过下载大小限制。")
                    output.write(chunk)
                return (b"" if path else output.getvalue()), str(response.url)
    raise SocialMediaError("分享链接跳转次数过多，请换成作品详情页链接。")


def _states(page: str) -> Iterator[dict]:
    """读取页面已提供的作品数据，不执行网页脚本，也不读取推荐流里的其他作品。"""
    from yt_dlp.utils import js_to_json

    for match in re.finditer(
        r"<script[^>]*\bid=[\"']RENDER_DATA[\"'][^>]*>(.*?)</script>", page, re.S,
    ):
        try:
            value = json.loads(unquote(html.unescape(match.group(1))))
            if isinstance(value, dict):
                yield value
        except ValueError:
            pass
    for match in re.finditer(r"(?:window\.)?(?:__INITIAL_STATE__|_ROUTER_DATA)\s*=\s*", page):
        try:
            value, _ = json.JSONDecoder().raw_decode(js_to_json(page[match.end():]))
            if isinstance(value, dict):
                yield value
        except (ValueError, TypeError):
            pass


def _find_item(value, item_id: str) -> dict | None:
    if isinstance(value, dict):
        if str(value.get("aweme_id") or value.get("awemeId") or
               value.get("noteId") or value.get("note_id") or "") == item_id:
            return value
        details = value.get("noteDetailMap")
        if isinstance(details, dict):
            detail = details.get(item_id)
            if isinstance(detail, dict) and isinstance(detail.get("note"), dict):
                return detail["note"]
        for nested in value.values():
            found = _find_item(nested, item_id)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_item(nested, item_id)
            if found:
                return found
    return None


def _clean_url(value) -> str | None:
    if not isinstance(value, str):
        return None
    url = html.unescape(value)
    if url.startswith("//"):
        url = f"https:{url}"
    elif url.startswith("http://"):
        url = f"https://{url.removeprefix('http://')}"
    try:
        clean = unquote(url).lower()
        # 只选平台无水印资源；不擦除图片里作者自己画上的文字、署名或 Logo。
        if _host_allowed(url, MEDIA_DOMAINS) and not any(
            marker in clean.replace("watermark=0", "").replace("watermark=false", "")
            for marker in ("/playwm/", "watermark", "wm=1")
        ):
            return url
    except ValueError:
        pass
    return None


def _image_urls(item: dict, platform: str) -> list[str]:
    # 小红书视频笔记的 imageList 只有封面，不能把封面当成整条视频来识别。
    if platform == "xiaohongshu" and item.get("type") == "video":
        return []
    images = item.get("images") if platform == "douyin" else item.get("imageList")
    if not isinstance(images, list):
        images = (item.get("image_post_info") or {}).get("images", [])
    urls = []
    for image in images:
        if not isinstance(image, dict):
            continue
        if platform == "douyin":
            candidates = image.get("url_list") or (image.get("display_image") or {}).get("url_list", [])
        else:
            # 小红书详情页的图片资源优先于封面、头像和页面截图。
            candidates = [
                image.get("urlDefault"), image.get("urlPre"), image.get("url"),
                *(info.get("url") for info in image.get("infoList", []) if isinstance(info, dict)),
            ]
        url = next((url for candidate in candidates if (url := _clean_url(candidate))), None)
        if url and url not in urls:
            urls.append(url)
    return urls


class _QuietDownloader:
    def debug(self, message) -> None:
        pass

    def warning(self, message) -> None:
        pass

    def error(self, message) -> None:
        pass


def _video_url(
    url: str, item: dict, client: httpx.Client, directory: Path,
) -> str:
    video = item.get("video") or {}
    addresses = (video.get("play_addr") or {}).get("url_list", [])
    for address in addresses:
        clean = _clean_url(address)
        if clean:
            return clean
    # 小红书详情页已经提供实际视频流，沿用同一下载、抽帧与域名限制。
    for variants in ((video.get("media") or {}).get("stream") or {}).values():
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            for address in [variant.get("masterUrl"), *(variant.get("backupUrls") or [])]:
                clean = _clean_url(address)
                if clean:
                    return clean

    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    cookie_path = directory / "cookies.txt"
    cookie_jar = MozillaCookieJar(str(cookie_path))
    for cookie in client.cookies.jar:
        cookie_jar.set_cookie(cookie)
    cookie_jar.save(ignore_discard=True, ignore_expires=True)
    cookie_path.chmod(0o600)
    try:
        with YoutubeDL({
            "quiet": True, "no_warnings": True, "logger": _QuietDownloader(),
            "noplaylist": True, "skip_download": True, "socket_timeout": 15,
            "retries": 1, "extractor_retries": 1, "cookiefile": str(cookie_path),
            "format": "best[ext=mp4]/best", "http_headers": {"User-Agent": USER_AGENT},
        }) as downloader:
            info = downloader.extract_info(url, download=False)
        clean = _clean_url(info.get("url")) if isinstance(info, dict) else None
        if clean:
            return clean
    except DownloadError as exc:
        raise SocialMediaError("视频解析失败，可能需要更新平台 Cookie，或作品已不可访问。") from exc
    raise SocialMediaError("没有拿到这条作品的无水印图片或视频地址。")


def _video_frames(path: Path, source_key: str) -> Iterator[SocialImage]:
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = (
                float(Decimal(container.duration) / Decimal(av.time_base))
                if container.duration else 0
            )
            if duration <= 0:
                frame = next(container.decode(stream), None)
                if frame is not None:
                    output = io.BytesIO()
                    frame.to_image().save(output, format="JPEG", quality=95)
                    yield SocialImage(f"{source_key}:frame:0", "视频帧", output.getvalue())
                return
            # 在全片均匀取十二个时间点，不能把封面命中当作视频内容已识别。
            for index, second in enumerate(np.linspace(0, duration, VIDEO_SAMPLES, endpoint=False)):
                container.seek(int(Decimal(str(second)) * Decimal(av.time_base)), backward=True)
                for frame in container.decode(stream):
                    if frame.time is not None and frame.time < second:
                        continue
                    output = io.BytesIO()
                    frame.to_image().save(output, format="JPEG", quality=95)
                    yield SocialImage(f"{source_key}:frame:{index}", "视频帧", output.getvalue())
                    break
    except (av.FFmpegError, OSError, IndexError, ValueError) as exc:
        raise SocialMediaError("下载的媒体无法读取视频画面。") from exc


def iter_social_images(
    url: str, *, douyin_cookie_file: Path | None = None,
    xiaohongshu_cookie_file: Path | None = None,
    share_text: str = "",
) -> Iterator[SocialImage]:
    """逐张下载并交给调用方识别，避免一次把整篇笔记的图片堆进内存。"""
    try:
        platform = "douyin" if "douyin.com" in (urlparse(url).hostname or "") else "xiaohongshu"
        cookie = douyin_cookie_file if platform == "douyin" else xiaohongshu_cookie_file
        root = "douyin.com" if platform == "douyin" else "xiaohongshu.com"
        with tempfile.TemporaryDirectory(prefix="milk-social-") as temporary:
            directory = Path(temporary)
            with httpx.Client(
                timeout=20, headers={"User-Agent": USER_AGENT, "Referer": f"https://www.{root}/"},
            ) as client:
                _load_cookies(client, cookie, root)
                content, resolved = _download(
                    client, url, domains=PAGE_DOMAINS, max_bytes=4 * 1024 * 1024,
                )
                if platform == "xiaohongshu" and urlparse(resolved).path.startswith("/login"):
                    raise SocialMediaError("小红书要求登录，请配置有效的小红书 Cookie 后重试。")
                pattern = (
                    r"/(?:video|note|share/video|share/slides)/(\d+)"
                    if platform == "douyin"
                    else r"/(?:explore|discovery/item|search_result)/([a-fA-F0-9]{24})"
                )
                parsed = urlparse(resolved)
                match = re.search(pattern, parsed.path)
                modal_id = (
                    parse_qs(parsed.query).get("modal_id", [""])[0]
                    if platform == "douyin" and parsed.path.startswith("/search/") else ""
                )
                if not match and not modal_id.isdigit():
                    raise SocialMediaError("请分享具体的抖音作品或小红书笔记链接，不支持个人主页。")
                item_id = match.group(1) if match else modal_id
                item = next(
                    (item for state in _states(content.decode("utf-8", errors="replace"))
                     if (item := _find_item(state, item_id))), {},
                )
                # 短链跳转后的详情页可能没有图片数据；用分享标题短前缀检索，
                # 再按原作品 ID 精确匹配，不能拿其他搜索结果代替用户分享的图文。
                caption = re.search(r"[\u4e00-\u9fff][^#\r\n。！？]*", share_text)
                keyword = (
                    unquote(parsed.path.removeprefix("/search/"))
                    if modal_id.isdigit() else caption.group()[:7] if caption else ""
                )
                if not item and platform == "douyin" and keyword:
                    params = urlencode({
                        "device_platform": "webapp", "aid": "6383", "channel": "channel_pc_web",
                        "keyword": keyword,
                        "search_channel": "aweme_general", "offset": "0", "count": "20",
                        "version_code": "190500", "version_name": "19.5.0",
                        "cookie_enabled": "true", "platform": "PC",
                    })
                    result, _ = _download(
                        client, "https://www.douyin.com/aweme/v1/web/general/search/single/"
                        f"?{params}", domains=PAGE_DOMAINS, max_bytes=4 * 1024 * 1024,
                    )
                    item = _find_item(json.loads(result), item_id) or {}
                if not item and platform == "douyin":
                    # 部分分享页不含内嵌作品数据，复用当前 Cookie 读取图文详情。
                    try:
                        detail, _ = _download(
                            client,
                            "https://www.douyin.com/aweme/v1/web/aweme/detail/"
                            f"?aid=6383&device_platform=webapp&aweme_id={item_id}",
                            domains=PAGE_DOMAINS, max_bytes=4 * 1024 * 1024,
                        )
                        item = _find_item(json.loads(detail), item_id) or {}
                    except (httpx.HTTPError, SocialMediaError, ValueError):
                        pass
                images = _image_urls(item, platform)
                source_key = f"{platform}:{item_id}"
                if images:
                    for index, image_url in enumerate(images[:MAX_IMAGES]):
                        try:
                            data, _ = _download(
                                client, image_url, domains=MEDIA_DOMAINS, max_bytes=MAX_IMAGE_BYTES,
                            )
                            with Image.open(io.BytesIO(data)) as image:
                                image.verify()
                        except (httpx.HTTPError, SocialMediaError, OSError, ValueError):
                            data = None
                        yield SocialImage(f"{source_key}:image:{index}", "图文图片", data)
                    return
                video_url = _video_url(url, item, client, directory)
                video_path = directory / "video.mp4"
                _download(
                    client, video_url, domains=MEDIA_DOMAINS,
                    max_bytes=MAX_VIDEO_BYTES, path=video_path,
                )
                yield from _video_frames(video_path, source_key)
                return
    except SocialMediaError:
        raise
    except ImportError as exc:
        raise SocialMediaError("服务器缺少媒体解析依赖，请更新机器人部署。") from exc
    except (httpx.HTTPError, OSError, ValueError, TypeError, RecursionError) as exc:
        raise SocialMediaError("链接解析或下载失败，请确认作品公开可见，必要时更新平台 Cookie。") from exc
