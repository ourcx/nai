from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_hdr_bot import run_mock_bot
from test_qq_adapter import event, image_bytes, mock_adapter, reply_text

from nailong_classifier.application import bot
from nailong_classifier.application.services import BackendServices, SerializedPredictor
from nailong_classifier.persistence.storage import BotStore


@pytest.mark.parametrize(
    "text,chat_type,mentioned,unknown",
    [
        ("help", "group", True, False),
        (" HELP ", "p2p", False, False),
        ("/help", "p2p", False, False),
        ("帮助", "p2p", False, False),
        ("纠正为奶牛", "group", True, True),
        ("猜图帮", "p2p", False, True),
        ("", "group", True, False),
    ],
)
def test_feishu_help_and_unknown_commands_reply_with_usage(
    tmp_path, monkeypatch, text, chat_type, mentioned, unknown
):
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text=text, chat_type=chat_type,
        mentioned_bot=mentioned, random_push_probability=0,
    )
    assert len(sent) == 1
    content, options = sent[0]
    assert "card" in content
    assert "奶龙图鉴" in content["fallback_text"]
    assert "私聊直接发" in content["fallback_text"]
    assert ("没认出这个指令" in content["fallback_text"]) is unknown
    assert options == {"reply_to": "source-id"}


@pytest.mark.parametrize(
    "text,chat_type,mentioned,expected",
    [
        ("奶龙表情 点赞", "p2p", False, "1000000000000000001"),
        ("奶蛙表情：奶龙 OK", "group", True, "1000000000000000002"),
        ("奶龙表情", "p2p", False, "list"),
        ("奶龙表情 点赞", "group", False, None),
    ],
)
def test_feishu_enterprise_emoji_commands(
    tmp_path, monkeypatch, text, chat_type, mentioned, expected
):
    mapping = tmp_path / "emoji-ids.json"
    mapping.write_text(
        '{"点赞":"1000000000000000001","OK":"1000000000000000002"}', encoding="utf-8"
    )
    monkeypatch.setenv("NAILONG_FEISHU_EMOJI_IDS_FILE", str(mapping))
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text=text, chat_type=chat_type,
        mentioned_bot=mentioned, random_push_probability=0,
    )
    if expected is None:
        assert not sent
        return
    content, options = sent[0]
    assert len(sent) == 1
    assert options == {"reply_to": "source-id"}
    if expected == "list":
        assert "点赞" in content["text"] and "OK" in content["text"]
    else:
        from lark_channel.channel._coerce import coerce_outbound
        from lark_channel.channel.outbound.sender import _build_post

        request = _build_post(coerce_outbound(content))
        assert request["msg_type"] == "post"
        assert json.loads(request["content"])["zh_cn"]["content"] == [
            [{"tag": "emotion", "emoji_type": expected}]
        ]


@pytest.mark.parametrize("text", ["help", "/help", "纠正为奶牛", "随便聊聊"])
def test_normal_feishu_group_chat_does_not_show_usage(tmp_path, monkeypatch, text):
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text=text, random_push_probability=0,
    )
    assert sent == []


def test_help_takes_priority_over_quoted_image_and_member_lookup(tmp_path, monkeypatch):
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="help", mentioned_bot=True, quoted_image=True,
        mentions=[SimpleNamespace(open_id="user-b", name="奶友乙", is_bot=False)],
        random_push_probability=0,
    )
    assert len(sent) == 1
    assert "可以这样玩" in sent[0][0]["fallback_text"]


@pytest.mark.parametrize("quoted", [False, True])
def test_unknown_text_with_image_keeps_image_workflow(tmp_path, monkeypatch, quoted):
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="识别一下", mentioned_bot=True,
        image=not quoted, quoted_image=quoted, random_push_probability=0,
    )
    assert len(sent) == 1
    assert "text" not in sent[0][0]
    assert "card" in sent[0][0]


def test_qq_help_unknown_commands_and_normal_chat(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    store = BotStore(tmp_path)
    predictor = SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.65))
    services = BackendServices(predictor, store, None)

    async def exercise():
        adapter, calls = await mock_adapter()

        async def connect_events():
            cases = [
                ("GROUP_AT_MESSAGE_CREATE", "help", False),
                ("GROUP_AT_MESSAGE_CREATE", "/help", False),
                ("GROUP_AT_MESSAGE_CREATE", "", False),
                ("GROUP_AT_MESSAGE_CREATE", "奶龙图签", True),
                ("C2C_MESSAGE_CREATE", "帮助", False),
                ("C2C_MESSAGE_CREATE", "HELP", False),
                ("C2C_MESSAGE_CREATE", "什么指令", True),
            ]
            for index, (event_type, text, unknown) in enumerate(cases):
                data = event(text, id=f"help-{index}")
                if event_type == "C2C_MESSAGE_CREATE":
                    data["content"] = text
                before = len(calls)
                await adapter.handle_event(event_type, data)
                messages = [p for path, p in calls[before:] if path.endswith("/messages")]
                assert len(messages) == 1
                assert messages[0]["msg_id"] == data["id"]
                assert "接奶（或奶家人）" in reply_text(messages[0])
                assert ("没认出这个指令" in reply_text(messages[0])) is unknown

            before = len(calls)
            for index, text in enumerate(["help", "纠正为奶牛", "随便聊聊"]):
                data = event(id=f"normal-{index}")
                data["content"] = text
                await adapter.handle_event("GROUP_MESSAGE_CREATE", data)
            assert len(calls) == before

        adapter.connect = connect_events
        try:
            await bot.run_adapter(args, adapter, services)
        finally:
            await adapter.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("text,chat_type,quoted", [
    ("随机奶图", "group", True),
    (" 来张奶图 ", "p2p", False),
    ("来张随机奶图", "group", False),
])
def test_feishu_random_milk_request_returns_one_plain_image(
    tmp_path, monkeypatch, text, chat_type, quoted,
):
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text=text, chat_type=chat_type, mentioned_bot=True,
        quoted_image=quoted, hdr_probability=1, random_push_probability=0,
        mentions=[SimpleNamespace(open_id="user-b", name="奶友乙", is_bot=False)],
    )
    assert len(sent) == 1
    content, options = sent[0]
    assert set(content) == {"image"}
    assert isinstance(content["image"]["source"], str)
    assert options == {"reply_to": "source-id"}


@pytest.mark.parametrize("seed_label", ["naiwa", "other"])
def test_qq_random_milk_request_uses_milk_gallery_and_handles_empty(
    tmp_path, monkeypatch, seed_label,
):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    store = BotStore(tmp_path)
    store.register_seed(image_bytes(), seed_label)
    predictor = SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.65))

    async def exercise():
        adapter, calls = await mock_adapter()

        async def connect_events():
            for index, event_type in enumerate(("GROUP_AT_MESSAGE_CREATE", "C2C_MESSAGE_CREATE")):
                data = event("随机奶图", id=f"random-{index}")
                if event_type == "C2C_MESSAGE_CREATE":
                    data["content"] = "随机奶图"
                await adapter.handle_event(event_type, data)

        adapter.connect = connect_events
        try:
            await bot.run_adapter(args, adapter, BackendServices(predictor, store, None))
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert len(messages) == 2
            assert [payload["msg_id"] for payload in messages] == ["random-0", "random-1"]
            if seed_label == "naiwa":
                assert all(payload["msg_type"] == 7 and "media" in payload for payload in messages)
            else:
                assert all("图库里还没有奶图" in reply_text(payload) for payload in messages)
        finally:
            await adapter.close()

    asyncio.run(exercise())
