from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from test_qq_adapter import event, image_bytes, mock_adapter

from nailong_classifier.adapters.base import ErrorCode
from nailong_classifier.adapters.qq import qq_plain_text


def test_qq_markdown_and_command_buttons_escape_body_and_remove_emoji():
    async def exercise():
        adapter, calls = await mock_adapter(custom_keyboard_enabled=True)
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            chat = adapter.scope.key("chat", "group:group-a")
            text = "今日奶运：奶友\n幸运物：🥛 牛奶\n[昵称](https://fake.test)\n稀有度：★★★☆☆"
            assert (await adapter.send(chat, {"card": {}, "fallback_text": text})).success
            payload = calls[-1][1]
            assert payload["msg_type"] == 2 and "content" not in payload
            assert payload["markdown"]["content"].startswith("# 今日奶运")
            assert "**幸运物：**" in payload["markdown"]["content"]
            assert r"\[昵称\]\(https://fake.test\)" in payload["markdown"]["content"]
            assert "3 星" in payload["markdown"]["content"]
            assert "🥛" not in str(payload)
            buttons = [
                button for row in payload["keyboard"]["content"]["rows"]
                for button in row["buttons"]
            ]
            assert [button["action"]["data"] for button in buttons] == [
                "今日奶运", "奶签收藏册", "奶签历史", "帮助"
            ]
            assert all(button["action"]["type"] == 2 and button["action"]["enter"]
                       for button in buttons)
            assert (await adapter.send(
                chat, {"image": {"source": image_bytes()}, "caption": "🥛 奶图"}
            )).success
            assert calls[-1][1]["msg_type"] == 7
            assert calls[-1][1]["content"] == "奶图"
            assert "keyboard" not in calls[-1][1]
        finally:
            await adapter.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("options,expected", [
    ({}, None),
    ({"keyboard_template_id": "approved-id"}, {"id": "approved-id"}),
])
def test_qq_default_does_not_assume_custom_button_permission(options, expected):
    async def exercise():
        adapter, calls = await mock_adapter(**options)
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            assert (await adapter.send(
                adapter.scope.key("chat", "group:group-a"), {"text": "奶签历史\n暂无"}
            )).success
            assert calls[-1][1].get("keyboard") == expected
        finally:
            await adapter.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("failure,expected_types,error_code", [
    ("rejected", [2, 2, 0], None),
    ("timeout", [2], ErrorCode.SEND_TIMEOUT),
    ("server", [2], ErrorCode.UNKNOWN),
])
def test_qq_style_falls_back_only_after_definite_rejection(failure, expected_types, error_code):
    payloads = []

    def api(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if failure == "timeout":
            raise httpx.ReadTimeout("uncertain result", request=request)
        if failure == "server":
            return httpx.Response(500, json={"code": 123})
        if payload["msg_type"] == 2:
            return httpx.Response(403, json={"code": 123})
        return httpx.Response(200, json={"id": "plain"})

    async def exercise():
        adapter, _ = await mock_adapter(api=api, custom_keyboard_enabled=True)
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            result = await adapter.send(
                adapter.scope.key("chat", "group:group-a"), {"text": "奶签历史\n暂无"}
            )
            assert [payload["msg_type"] for payload in payloads] == expected_types
            assert all(payload["msg_seq"] == 1 for payload in payloads)
            if error_code:
                assert not result.success and result.error.code == error_code
            else:
                assert result.success
                assert "keyboard" in payloads[0] and "keyboard" not in payloads[1]
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_qq_plain_mode_strips_emoji_sequences_and_preserves_numbers():
    assert qq_plain_text("👨‍👩‍👧‍👦 🥛 1️⃣ 🇨🇳 123 ★★★☆☆") == "123 3 星"


def test_qq_fortune_collection_buttons_use_business_commands(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nailong_classifier.application import bot
    from nailong_classifier.application.services import BackendServices, SerializedPredictor
    from nailong_classifier.persistence.storage import BotStore

    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    store = BotStore(tmp_path)

    async def exercise():
        adapter, calls = await mock_adapter(custom_keyboard_enabled=True)

        async def connect_events():
            for index, command in enumerate(("今日奶运", "奶签收藏册", "奶签历史")):
                await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event(command, id=str(index)))

        monkeypatch.setattr(adapter, "connect", connect_events)
        try:
            await bot.run_adapter(args, adapter, BackendServices(
                SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.65)), store,
            ))
            payloads = [payload for path, payload in calls if path.endswith("/messages")]
            assert len(payloads) == 3
            assert all(payload["msg_type"] == 2 for payload in payloads)
            assert any(label in payloads[0]["markdown"]["content"]
                       for label in ("鸡腿运", "荷塘运", "深潜运", "嘴甜运"))
            assert "签面 1/128" in payloads[1]["markdown"]["content"]
            assert "第 1/1 页" in payloads[2]["markdown"]["content"]
        finally:
            await adapter.close()
    asyncio.run(exercise())
