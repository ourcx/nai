from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from test_hdr_bot import run_mock_bot
from test_qq_adapter import event, mock_adapter

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
    assert "奶龙图鉴" in content["text"]
    assert "私聊直接发" in content["text"]
    assert ("没认出这个指令" in content["text"]) is unknown
    assert options == {"reply_to": "source-id"}


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
    assert "可以这样玩" in sent[0][0]["text"]


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
                assert "接奶（或奶家人）" in messages[0]["content"]
                assert ("没认出这个指令" in messages[0]["content"]) is unknown

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
