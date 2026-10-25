from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import imagecodecs
import pytest
from lark_channel.channel.errors import FeishuChannelErrorCode
from PIL import Image

import nailong_classifier.bot as bot
from nailong_classifier.douyin import DouyinVideo
from nailong_classifier.model import Prediction
from nailong_classifier.storage import BotStore


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("text", ["HDR奶图", "hdr奶图", " HDR奶龙 ", "来张HDR奶图"])
def test_hdr_command_matches_explicit_requests(text: str) -> None:
    assert bot.is_hdr_milk_command(text)


@pytest.mark.parametrize("text", ["奶龙", "猜图", "hdr", "HDR奶图很好笑"])
def test_hdr_command_does_not_match_normal_chat(text: str) -> None:
    assert not bot.is_hdr_milk_command(text)


def run_mock_bot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str,
    image: bool = False,
    hdr_probability: float = 1.0,
    douyin_cookie_file: Path | None = None,
    random_push_probability: float = 1.0,
    mentioned_bot: bool = False,
    first_error: FeishuChannelErrorCode | None = None,
):
    store = BotStore(tmp_path)
    store.register_seed(image_bytes("yellow"), "nailong")
    calls = []

    class Channel:
        client = SimpleNamespace()

        def __init__(self, **kwargs):
            self.handlers = {}

        def on(self, event, handler):
            self.handlers[event] = handler

        async def send(self, chat_id, content, options=None):
            calls.append((content, options))
            if len(calls) == 1 and first_error is not None:
                return SimpleNamespace(
                    success=False, error=SimpleNamespace(code=first_error), message_id=None
                )
            return SimpleNamespace(success=True, error=None, message_id="response-id")

        async def connect(self):
            message = SimpleNamespace(
                sender_is_bot=False,
                raw_content_type="image" if image else "text",
                resources=[SimpleNamespace(type="image", file_key="img_source")] if image else [],
                reply_to_message_id=None,
                mentioned_bot=mentioned_bot,
                message_id="source-id",
                chat_id="chat-a",
                chat_type="group",
                sender_id="user-a",
                sender_name="奶友",
                body_text=text,
                content_text=text,
            )
            await self.handlers["message"](message)

    async def download(*args):
        return image_bytes("blue")

    predictor = SimpleNamespace(
        device="cpu",
        threshold=0.65,
        predict_bytes=lambda data: Prediction(
            label="nailong",
            confidence=0.9,
            accepted=True,
            scores={"nailong": 0.9, "naiwa": 0.05, "other": 0.05},
        ),
    )
    monkeypatch.setenv("LARK_APP_ID", "test-app")
    monkeypatch.setenv("LARK_APP_SECRET", "test-secret")
    monkeypatch.delenv("CATALOG_DOCUMENT_ID", raising=False)
    monkeypatch.delenv("CATALOG_SECTION_BLOCK_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["nailong-bot"])
    monkeypatch.setattr(bot, "FeishuChannel", Channel)
    monkeypatch.setattr(bot, "NailongPredictor", lambda *args, **kwargs: predictor)
    monkeypatch.setattr(bot, "download_target", download)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.random_push_probability = random_push_probability
    args.random_push_hdr_probability = hdr_probability
    args.douyin_cookie_file = douyin_cookie_file
    args.douyin_video_push_probability = 1
    asyncio.run(bot.run_bot(args))
    return calls, store


def test_manual_hdr_sends_image_even_when_random_hdr_is_disabled(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(tmp_path, monkeypatch, text="HDR奶图", hdr_probability=0)
    assert len(calls) == 1
    content, options = calls[0]
    assert set(content) == {"image"}
    assert options == {"reply_to": "source-id"}
    assert imagecodecs.ultrahdr_check(content["image"]["source"])


def test_chain_reply_runs_hdr_conversion_and_preserves_response_tracking(tmp_path, monkeypatch):
    calls, store = run_mock_bot(tmp_path, monkeypatch, text="", image=True)
    assert len(calls) == 1
    content, options = calls[0]
    assert set(content) == {"image"}
    assert imagecodecs.ultrahdr_check(content["image"]["source"])
    assert options == {"reply_to": "source-id"}
    correction = store.correct_label("response-id", "naiwa")
    assert correction is not None


def test_chain_without_hdr_keeps_existing_caption(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(tmp_path, monkeypatch, text="", image=True, hdr_probability=0)
    content, _ = calls[0]
    assert isinstance(content["image"]["source"], str)
    assert "奶龙接上了" in content["caption"]


def test_normal_chat_random_push_uses_shared_hdr_sender(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(tmp_path, monkeypatch, text="随便聊聊")
    assert len(calls) == 1
    content, options = calls[0]
    assert set(content) == {"image"}
    assert options is None
    assert imagecodecs.ultrahdr_check(content["image"]["source"])


def test_normal_chat_does_not_send_unverified_static_douyin_urls(
    tmp_path, monkeypatch
) -> None:
    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="随便聊聊",
        random_push_probability=0,
    )

    assert calls == []


def test_normal_chat_searches_douyin_when_cookie_search_is_enabled(tmp_path, monkeypatch) -> None:
    cookie_file = tmp_path / "douyin-cookies.json"
    cookie_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(bot, "download_douyin_cover", lambda url: b"cover")
    monkeypatch.setattr(
        bot,
        "search_douyin_videos",
        lambda path, *, keywords: [
            DouyinVideo(
                url="https://www.douyin.com/video/123",
                title="奶蛙测试视频",
                keyword="奶蛙",
                author_name="奶友",
                cover_url="https://p3.douyinpic.com/test-cover.webp",
            )
        ],
    )

    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="随便聊聊",
        douyin_cookie_file=cookie_file,
    )

    assert calls == [
        ({"text": "来点奶：https://www.douyin.com/video/123"}, None),
    ]


def test_mentioned_topic_search_replies_with_requested_topic(tmp_path, monkeypatch) -> None:
    cookie_file = tmp_path / "douyin-cookies.json"
    cookie_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(bot, "download_douyin_cover", lambda url: b"cover")
    monkeypatch.setattr(
        bot,
        "search_douyin_videos",
        lambda path, *, keywords: [
            DouyinVideo(
                url="https://www.douyin.com/video/456",
                title="豆奶测试视频",
                keyword="豆奶",
                author_name="奶友",
                cover_url="https://p3.douyinpic.com/test-cover.webp",
            )
        ],
    )

    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="#豆奶",
        douyin_cookie_file=cookie_file,
        mentioned_bot=True,
    )

    assert calls == [
        ({"text": "来点#豆奶：https://www.douyin.com/video/456"}, {"reply_to": "source-id"}),
    ]


def test_douyin_cover_filter_requires_accepted_milk_prediction(monkeypatch) -> None:
    videos = [
        DouyinVideo(
            url="https://www.douyin.com/video/keep",
            title="奶蛙视频",
            keyword="奶蛙",
            author_name="奶友甲",
            cover_url="https://p3.douyinpic.com/keep.webp",
        ),
        DouyinVideo(
            url="https://www.douyin.com/video/drop",
            title="无关视频",
            keyword="奶蛙",
            author_name="奶友乙",
            cover_url="https://p3.douyinpic.com/drop.webp",
        ),
        DouyinVideo(
            url="https://www.douyin.com/video/no-cover",
            title="无封面视频",
            keyword="奶蛙",
            author_name="奶友丙",
        ),
    ]
    monkeypatch.setattr(bot, "download_douyin_cover", lambda url: url.encode())

    def predict_bytes(data: bytes) -> Prediction:
        return Prediction(
            label="naiwa" if data.endswith(b"keep.webp") else "other",
            confidence=0.9,
            accepted=True,
            scores={"nailong": 0.05, "naiwa": 0.9, "other": 0.05},
        )

    predictor = SimpleNamespace(predict_bytes=predict_bytes)

    assert bot.filter_douyin_videos_by_cover(videos, predictor) == [videos[0]]


def test_hdr_upload_rejection_falls_back_once_to_original(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(
        tmp_path, monkeypatch, text="HDR奶图", first_error=FeishuChannelErrorCode.UPLOAD_FAILED
    )
    assert len(calls) == 2
    content, options = calls[1]
    assert set(content) == {"image"}
    assert isinstance(content["image"]["source"], str)
    assert options == {"reply_to": "source-id"}


def test_revoked_chat_does_not_trigger_fallback_send(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(
        tmp_path, monkeypatch, text="HDR奶图", first_error=FeishuChannelErrorCode.TARGET_REVOKED
    )
    assert len(calls) == 1
