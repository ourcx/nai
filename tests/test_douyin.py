from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nailong_classifier.douyin import DouyinSearchError, search_douyin_videos


class FakeResponse(io.BytesIO):
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
                        {"aweme_id": "123", "desc": f"{keyword}第一条"},
                        {"aweme_id": "123", "desc": f"{keyword}重复条"},
                        {"aweme_id": "456", "desc": "无关内容"},
                    ]
                },
                {"aweme_info": {"aweme_id": "789", "desc": f"{keyword}卡片视频"}},
            ],
        }
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("nailong_classifier.douyin.urlopen", fake_urlopen)

    videos = search_douyin_videos(cookie_file, count=3)

    assert [video.url for video in videos] == [
        "https://www.douyin.com/video/123",
        "https://www.douyin.com/video/789",
    ]
    assert len(requests) == 2
    assert requests[0].get_header("Cookie") == "sessionid=private-cookie-value"


def test_search_douyin_videos_rejects_invalid_cookie_file(tmp_path: Path) -> None:
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text("{}", encoding="utf-8")

    with pytest.raises(DouyinSearchError, match="JSON list"):
        search_douyin_videos(cookie_file)
