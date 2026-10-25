from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nailong_classifier.douyin import (
    DouyinSearchError,
    DouyinVideo,
    choose_douyin_video,
    download_douyin_cover,
    search_douyin_videos,
)


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(data)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_cookies(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": "sessionid",
                    "value": "private-cookie-value",
                    "domain": ".douyin.com",
                    "path": "/",
                },
                {
                    "name": "ignored",
                    "value": "wrong-domain",
                    "domain": ".example.com",
                    "path": "/",
                },
            ]
        ),
        encoding="utf-8",
    )


def test_search_douyin_videos_filters_by_keyword_and_deduplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cookie_file = tmp_path / "cookies.json"
    write_cookies(cookie_file)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        keyword = "奶龙" if "%E5%A5%B6%E9%BE%99" in request.full_url else "奶娃"
        payload = {
            "status_code": 0,
            "data": [
                {
                    "aweme_list": [
                        {
                            "aweme_id": "123",
                            "desc": f"{keyword}第一条",
                            "author": {"nickname": "奶友"},
                            "cha_list": [{"cha_name": keyword}],
                            "video": {
                                "cover": {
                                    "url_list": [
                                        f"https://p3.douyinpic.com/{keyword}-cover.webp"
                                    ]
                                }
                            },
                        },
                        {
                            "aweme_id": "123",
                            "desc": f"{keyword}重复条",
                            "author": {"nickname": "奶友"},
                            "cha_list": [{"cha_name": keyword}],
                            "video": {
                                "cover": {
                                    "url_list": [
                                        f"https://p3.douyinpic.com/{keyword}-duplicate.webp"
                                    ]
                                }
                            },
                        },
                        {"aweme_id": "456", "desc": "无关内容"},
                    ]
                },
                {
                    "aweme_info": {
                        "aweme_id": "789",
                        "desc": f"{keyword}卡片视频",
                        "author": {"nickname": "奶友"},
                        "text_extra": [{"hashtag_name": f"#{keyword}"}],
                        "video": {
                            "origin_cover": {
                                "url_list": [f"https://p3.douyinpic.com/{keyword}-origin.webp"]
                            }
                        },
                    }
                },
                {
                    "aweme_info": {
                        "aweme_id": "998",
                        "desc": f"{keyword}标题含奶但没有标签",
                        "author": {"nickname": "奶友"},
                        "video": {
                            "cover": {
                                "url_list": [f"https://p3.douyinpic.com/{keyword}-untagged.webp"]
                            }
                        },
                    }
                },
                {
                    "aweme_info": {
                        "aweme_id": "999",
                        "desc": f"{keyword}官方视频",
                        "author": {"nickname": "奶龙"},
                        "cha_list": [{"cha_name": keyword}],
                    }
                },
            ],
        }
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("nailong_classifier.douyin.urlopen", fake_urlopen)

    videos = search_douyin_videos(cookie_file, keywords=("奶龙", "奶娃"), count=3)

    assert [video.url for video in videos] == [
        "https://www.douyin.com/video/123",
        "https://www.douyin.com/video/789",
    ]
    assert len(requests) == 2
    assert requests[0].get_header("Cookie") == "sessionid=private-cookie-value"
    assert videos[0].cover_url == "https://p3.douyinpic.com/奶龙-cover.webp"
    assert videos[1].cover_url == "https://p3.douyinpic.com/奶龙-origin.webp"
    assert videos[0].tags == ("奶龙",)
    assert videos[1].tags == ("奶龙",)


def test_search_douyin_videos_requires_a_structured_milk_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cookie_file = tmp_path / "cookies.json"
    write_cookies(cookie_file)
    payload = {
        "status_code": 0,
        "data": [
            {
                "aweme_info": {
                    "aweme_id": "123",
                    "desc": "文艺复兴视频",
                    "author": {"nickname": "奶友"},
                    "cha_list": [{"cha_name": "文艺复兴"}, {"cha_name": "奶蛙"}],
                }
            },
            {
                "aweme_info": {
                    "aweme_id": "456",
                    "desc": "文艺复兴标题里有奶龙",
                    "author": {"nickname": "奶友"},
                    "cha_list": [{"cha_name": "文艺复兴"}],
                }
            },
        ],
    }
    monkeypatch.setattr(
        "nailong_classifier.douyin.urlopen",
        lambda request, timeout: FakeResponse(json.dumps(payload).encode()),
    )

    videos = search_douyin_videos(cookie_file, keywords=("文艺复兴",))

    assert [video.url for video in videos] == ["https://www.douyin.com/video/123"]
    assert videos[0].tags == ("文艺复兴", "奶蛙")


def test_choose_douyin_video_selects_from_available_topics_equally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = [
        DouyinVideo(
            url="https://www.douyin.com/video/1",
            title="奶龙视频",
            keyword="奶龙",
            author_name="奶友甲",
        ),
        DouyinVideo(
            url="https://www.douyin.com/video/2",
            title="奶娃视频",
            keyword="奶娃",
            author_name="奶友乙",
        ),
    ]
    monkeypatch.setattr("nailong_classifier.douyin.secrets.choice", lambda values: values[0])

    assert choose_douyin_video(videos, keywords=("奶龙", "奶娃")).url == (
        "https://www.douyin.com/video/1"
    )


def test_search_douyin_videos_rejects_invalid_cookie_file(tmp_path: Path) -> None:
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text("{}", encoding="utf-8")

    with pytest.raises(DouyinSearchError, match="JSON list"):
        search_douyin_videos(cookie_file)


def test_download_douyin_cover_rejects_oversized_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nailong_classifier.douyin.urlopen",
        lambda request, timeout: FakeResponse(b"image", {"Content-Length": "6"}),
    )

    with pytest.raises(DouyinSearchError, match="maximum size"):
        download_douyin_cover("https://p3.douyinpic.com/cover.webp", max_bytes=5)
