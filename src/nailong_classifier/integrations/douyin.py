from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

SEARCH_ENDPOINT = "https://www.douyin.com/aweme/v1/web/general/search/single/"
SEARCH_KEYWORDS = ("奶蛙", "文艺复兴", "豆奶")
COOKIE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
OFFICIAL_AUTHOR_NAMES = frozenset({"奶龙"})
MILK_TAGS = frozenset({"奶龙", "奶蛙"})
MAX_COVER_BYTES = 5 * 1024 * 1024


class DouyinSearchError(RuntimeError):
    """抖音搜索未返回可用公开视频时抛出。"""


@dataclass(frozen=True)
class DouyinVideo:
    url: str
    title: str
    keyword: str
    author_name: str
    cover_url: str | None = None
    tags: tuple[str, ...] = ()


def _cookie_header(cookie_file: Path) -> str:
    """从浏览器导出的 Cookie JSON 读取抖音域名 Cookie，不回显任何 Cookie 内容。"""
    try:
        raw_cookies = json.loads(cookie_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DouyinSearchError("failed to read Douyin cookie file") from error
    if not isinstance(raw_cookies, list):
        raise DouyinSearchError("Douyin cookie file must contain a JSON list")

    pairs: list[str] = []
    for item in raw_cookies:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        domain = str(item.get("domain", "")).lstrip(".").lower()
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not COOKIE_NAME_PATTERN.fullmatch(name)
            or domain != "douyin.com"
            or "\r" in value
            or "\n" in value
        ):
            continue
        pairs.append(f"{name}={value}")
    if not pairs:
        raise DouyinSearchError("Douyin cookie file has no usable douyin.com cookies")
    return "; ".join(pairs)


def _videos_from_payload(payload: dict[str, Any], keyword: str) -> list[DouyinVideo]:
    videos: list[DouyinVideo] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        awemes = item.get("aweme_list")
        if not isinstance(awemes, list):
            aweme_info = item.get("aweme_info")
            awemes = [aweme_info] if isinstance(aweme_info, dict) else []
        for aweme in awemes:
            if not isinstance(aweme, dict):
                continue
            aweme_id = str(aweme.get("aweme_id", ""))
            title = str(aweme.get("desc", "")).strip()
            author = aweme.get("author")
            author_name = (
                str(author.get("nickname", "")).strip() if isinstance(author, dict) else ""
            )
            tags = _tags_from_aweme(aweme)
            if (
                not aweme_id.isdigit()
                or keyword not in title
                or not any(tag in MILK_TAGS for tag in tags)
            ):
                continue
            if author_name in OFFICIAL_AUTHOR_NAMES:
                continue
            videos.append(
                DouyinVideo(
                    url=f"https://www.douyin.com/video/{aweme_id}",
                    title=title,
                    keyword=keyword,
                    author_name=author_name,
                    cover_url=_cover_url_from_aweme(aweme),
                    tags=tags,
                )
            )
    return videos


def _tags_from_aweme(aweme: dict[str, Any]) -> tuple[str, ...]:
    """读取抖音结构化话题字段，不从标题或普通文本猜测标签。"""
    tags: list[str] = []
    challenge_list = aweme.get("cha_list")
    if isinstance(challenge_list, list):
        for challenge in challenge_list:
            if isinstance(challenge, dict):
                tag = challenge.get("cha_name")
                if isinstance(tag, str) and tag.strip():
                    tags.append(tag.strip().lstrip("#"))
    text_extra = aweme.get("text_extra")
    if isinstance(text_extra, list):
        for extra in text_extra:
            if isinstance(extra, dict):
                tag = extra.get("hashtag_name")
                if isinstance(tag, str) and tag.strip():
                    tags.append(tag.strip().lstrip("#"))
    return tuple(dict.fromkeys(tags))


def _cover_url_from_aweme(aweme: dict[str, Any]) -> str | None:
    video = aweme.get("video")
    if not isinstance(video, dict):
        return None
    for cover_key in ("origin_cover", "cover", "dynamic_cover"):
        cover = video.get(cover_key)
        if not isinstance(cover, dict):
            continue
        urls = cover.get("url_list")
        if not isinstance(urls, list):
            continue
        for url in urls:
            if isinstance(url, str) and _is_https_url(url):
                return url
    return None


def _is_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.hostname)


def download_douyin_cover(
    cover_url: str,
    *,
    timeout_seconds: float = 10,
    max_bytes: int = MAX_COVER_BYTES,
) -> bytes:
    """下载搜索结果封面供本地分类，不跟随视频页且限制响应体大小。"""
    if not _is_https_url(cover_url):
        raise DouyinSearchError("Douyin cover URL must be HTTPS")
    request = Request(
        cover_url,
        headers={
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://www.douyin.com/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            ),
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise DouyinSearchError("Douyin cover exceeds maximum size")
            image_bytes = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        raise DouyinSearchError("Douyin cover download failed") from error
    if not image_bytes or len(image_bytes) > max_bytes:
        raise DouyinSearchError("Douyin cover exceeds maximum size")
    return image_bytes


def search_douyin_videos(
    cookie_file: Path,
    *,
    keywords: tuple[str, ...] = SEARCH_KEYWORDS,
    count: int = 10,
    timeout_seconds: float = 10,
) -> list[DouyinVideo]:
    """搜索关键词并返回可公开访问的视频详情页，不下载视频或抓取评论。"""
    if count <= 0:
        raise ValueError("count must be positive")
    cookie_header = _cookie_header(cookie_file)
    videos: dict[str, DouyinVideo] = {}

    for keyword in keywords:
        params = urlencode(
            {
                "device_platform": "webapp",
                "aid": "6383",
                "channel": "channel_pc_web",
                "keyword": keyword,
                "search_channel": "aweme_general",
                "offset": "0",
                "count": str(count),
                "version_code": "190500",
                "version_name": "19.5.0",
                "cookie_enabled": "true",
                "platform": "PC",
            }
        )
        request = Request(
            f"{SEARCH_ENDPOINT}?{params}",
            headers={
                "Accept": "application/json, text/plain, */*",
                "Cookie": cookie_header,
                "Referer": f"https://www.douyin.com/search/{quote(keyword)}",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0 Safari/537.36"
                ),
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise DouyinSearchError("Douyin search request failed") from error
        if not isinstance(payload, dict) or payload.get("status_code") != 0:
            raise DouyinSearchError("Douyin search returned an unsuccessful response")
        for video in _videos_from_payload(payload, keyword):
            videos.setdefault(video.url, video)

    if not videos:
        raise DouyinSearchError("Douyin search returned no matching videos")
    return list(videos.values())


def choose_douyin_video(
    videos: list[DouyinVideo],
    *,
    keywords: tuple[str, ...] = SEARCH_KEYWORDS,
) -> DouyinVideo:
    """先在有结果的话题之间等概率选择，再随机挑该话题的一条视频。"""
    by_keyword = {
        keyword: [video for video in videos if video.keyword == keyword]
        for keyword in keywords
    }
    available_keywords = [keyword for keyword in keywords if by_keyword[keyword]]
    if not available_keywords:
        raise DouyinSearchError("Douyin search has no video for configured topics")
    return secrets.choice(by_keyword[secrets.choice(available_keywords)])
