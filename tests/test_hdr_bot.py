from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from lark_channel.channel.errors import FeishuChannelErrorCode
from PIL import Image

import nailong_classifier.adapters.feishu as feishu
import nailong_classifier.bot as bot
from nailong_classifier.adapters.config import create_adapter
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
    mentions: list[object] | None = None,
    quoted_image: bool = False,
    first_error: FeishuChannelErrorCode | None = None,
    doubao_detected: bool = False,
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

        def on_raw_event(self, event_type, handler):
            self.handlers[f"raw:{event_type}"] = handler

        async def resolve_quoted_contexts(self, messages, **kwargs):
            if not quoted_image:
                return {}
            return {
                messages[0].message_id: SimpleNamespace(
                    decision="resolved",
                    context=SimpleNamespace(
                        content_type="image",
                        message_id="quoted-image-id",
                        raw={"body": {"content": '{"image_key":"quoted-image-key"}'}},
                    ),
                )
            }

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
                reply_to_message_id="quoted-image-id" if quoted_image else None,
                mentioned_bot=mentioned_bot,
                message_id="source-id",
                chat_id="chat-a",
                chat_type="group",
                sender_id="user-a",
                sender_name="奶友",
                mentions=mentions or [],
                body_text=text,
                content_text=text,
            )
            await self.handlers["message"](message)

        async def disconnect(self):
            pass

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
    monkeypatch.delenv("DOUBAO_MODEL_PATH", raising=False)
    monkeypatch.delenv("DOUBAO_MODEL_THRESHOLD", raising=False)
    monkeypatch.setattr(sys, "argv", ["nailong-bot"])
    monkeypatch.setattr(feishu, "FeishuChannel", Channel)
    # 这些业务回归使用进程内假 SDK；真实进程边界由 test_adapters 覆盖，
    # 不能把未打补丁的子进程连接到真实飞书网络。
    monkeypatch.setattr(
        bot, "create_adapter",
        lambda config, **kwargs: create_adapter(config, isolate_feishu=False),
    )
    monkeypatch.setattr(bot, "NailongPredictor", lambda *args, **kwargs: predictor)
    monkeypatch.setattr(bot, "download_target", download)
    if doubao_detected:
        doubao_detector = SimpleNamespace(
            threshold=0.3,
            predict_bytes=lambda data: SimpleNamespace(detected=True, confidence=0.9),
            predict_path=lambda path: SimpleNamespace(detected=True, confidence=0.9),
        )
        monkeypatch.setattr(bot, "DoubaoDetector", lambda *args, **kwargs: doubao_detector)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.random_push_probability = random_push_probability
    args.random_push_hdr_probability = hdr_probability
    args.douyin_cookie_file = douyin_cookie_file
    args.douyin_video_push_probability = 1
    args.doubao_model = Path("doubao-detector.pt") if doubao_detected else None
    asyncio.run(bot.run_bot(args))
    return calls, store


def _assert_flash_gif(source: bytes) -> None:
    with Image.open(io.BytesIO(source)) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3


def test_manual_hdr_sends_flash_gif_even_when_random_hdr_is_disabled(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(tmp_path, monkeypatch, text="HDR奶图", hdr_probability=0)
    assert len(calls) == 1
    content, options = calls[0]
    assert set(content) == {"image"}
    assert options == {"reply_to": "source-id"}
    _assert_flash_gif(content["image"]["source"])


def test_chain_reply_runs_flash_conversion_and_preserves_response_tracking(tmp_path, monkeypatch):
    calls, store = run_mock_bot(tmp_path, monkeypatch, text="", image=True)
    assert len(calls) == 1
    content, options = calls[0]
    assert set(content) == {"image"}
    _assert_flash_gif(content["image"]["source"])
    assert options == {"reply_to": "source-id"}
    correction = store.correct_label("response-id", "naiwa")
    assert correction is not None


def test_chain_without_hdr_keeps_existing_caption(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(tmp_path, monkeypatch, text="", image=True, hdr_probability=0)
    content, _ = calls[0]
    assert isinstance(content["image"]["source"], str)
    assert "奶龙接上了" in content["caption"]


def test_normal_chat_random_push_uses_shared_flash_sender(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(tmp_path, monkeypatch, text="随便聊聊")
    assert len(calls) == 1
    content, options = calls[0]
    assert set(content) == {"image"}
    assert options is None
    _assert_flash_gif(content["image"]["source"])


def test_mentioning_a_registered_milk_family_member_returns_public_milk_name(
    tmp_path, monkeypatch
) -> None:
    store = BotStore(tmp_path)
    store.record_milk_family_member("user-b", display_name="奶小王")

    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="@奶小王",
        mentioned_bot=True,
        mentions=[
            SimpleNamespace(open_id="bot-id", name="我是奶龙", is_bot=True),
            SimpleNamespace(open_id="user-b", name="奶小王", is_bot=False),
        ],
        random_push_probability=0,
    )

    assert len(calls) == 1
    content, options = calls[0]
    assert content["text"].startswith("奶小王 是奶家人，是你【")
    assert options == {"reply_to": "source-id"}


def test_mentioning_an_unregistered_user_explains_how_to_join(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="@奶新人",
        mentioned_bot=True,
        mentions=[SimpleNamespace(open_id="user-new", name="奶新人", is_bot=False)],
        random_push_probability=0,
    )

    assert calls == [
        (
            {"text": "奶新人 还没被我认到是奶家人，先在群里冒个泡吧。"},
            {"reply_to": "source-id"},
        )
    ]


def test_replying_to_an_image_and_mentioning_bot_still_runs_recognition(
    tmp_path, monkeypatch
) -> None:
    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="",
        mentioned_bot=True,
        mentions=[SimpleNamespace(open_id="bot-id", name="我是奶龙", is_bot=True)],
        quoted_image=True,
        random_push_probability=0,
    )

    assert len(calls) == 1
    content, options = calls[0]
    assert "card" in content
    assert options == {"reply_to": "source-id"}


def test_mentioning_bot_in_same_message_as_image_still_runs_recognition(
    tmp_path, monkeypatch
) -> None:
    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="@我是奶龙",
        image=True,
        mentioned_bot=True,
        mentions=[SimpleNamespace(open_id="bot-id", name="我是奶龙", is_bot=True)],
        random_push_probability=0,
    )

    assert len(calls) == 1
    content, options = calls[0]
    assert "card" in content
    assert options == {"reply_to": "source-id"}


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


def test_doubao_image_is_silently_rejected_before_collection(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="",
        image=True,
        doubao_detected=True,
    )

    assert calls == []


def test_doubao_asset_is_not_sent_by_random_push(tmp_path, monkeypatch) -> None:
    calls, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="正常聊天",
        doubao_detected=True,
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
