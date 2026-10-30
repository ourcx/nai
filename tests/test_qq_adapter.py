from __future__ import annotations

import asyncio
import base64
import io
import itertools
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve

import nailong_classifier.bot as bot
from nailong_classifier.adapters.base import ErrorCode, IdentityScope
from nailong_classifier.adapters.config import AdapterConfig, create_adapter, load_adapter_configs
from nailong_classifier.adapters.qq import QQAdapter
from nailong_classifier.model import Prediction
from nailong_classifier.services import BackendServices, SerializedPredictor
from nailong_classifier.storage import BotStore


def image_bytes(format="PNG", *, color="yellow") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(output, format=format)
    return output.getvalue()


def event(text="", *, id="source", group="group-a", image=False, parent=None):
    data = {
        "id": id,
        "content": f"<@!102498836> {text}",
        "author": {"member_openid": "member-a", "user_openid": "user-a"},
        "group_openid": group,
    }
    if image:
        data["attachments"] = [
            {"content_type": "image/png", "url": "//cdn.test/image.png"}
        ]
    if parent:
        data["message_reference"] = {"message_id": parent}
    return data


async def mock_adapter(*, api=None, auth=None, **kwargs):
    adapter = QQAdapter(
        IdentityScope("qq", "qq-main"), app_id="102498836", app_secret="dummy-secret", **kwargs
    )
    await adapter._api.aclose()
    await adapter._auth.aclose()
    await adapter._media.aclose()
    calls = []
    ids = itertools.count(1)

    def default_api(request):
        assert request.headers["Authorization"] == "QQBot test-token"
        data = json.loads(request.content) if request.content else None
        calls.append((request.url.path, data))
        result = (
            {"file_info": "test-file"} if request.url.path.endswith("/files")
            else {"id": f"reply-{next(ids)}"}
        )
        return httpx.Response(200, json=result)

    def default_auth(request):
        assert "Authorization" not in request.headers
        assert json.loads(request.content) == {
            "appId": "102498836", "clientSecret": "dummy-secret"
        }
        return httpx.Response(200, json={"access_token": "test-token", "expires_in": "7200"})

    def media(request):
        assert "Authorization" not in request.headers
        return httpx.Response(200, content=image_bytes())

    adapter._api = httpx.AsyncClient(
        base_url="https://api.bot.qq.com/", transport=httpx.MockTransport(api or default_api)
    )
    adapter._auth = httpx.AsyncClient(
        base_url="https://api.bot.qq.com/", transport=httpx.MockTransport(auth or default_auth)
    )
    adapter._media = httpx.AsyncClient(transport=httpx.MockTransport(media))
    return adapter, calls


def test_selected_official_qq_bot_can_be_created_from_config(tmp_path, monkeypatch):
    path = tmp_path / "bots.toml"
    path.write_text(
        '[[bots]]\nid="qq-main"\nplatform="qq"\napp_id="102498836"\n'
        'app_secret_env="QQ_APP_SECRET"\n'
    )
    monkeypatch.delenv("QQ_APP_SECRET", raising=False)
    config = load_adapter_configs(path)[0]
    with pytest.raises(ValueError, match="environment variable"):
        create_adapter(config)
    monkeypatch.setenv("QQ_APP_SECRET", "test-secret")

    async def exercise():
        adapter = create_adapter(config)
        try:
            assert isinstance(adapter, QQAdapter)
            assert adapter.app_id == "102498836"
            assert not adapter.allow_proactive
        finally:
            await adapter.close()
    asyncio.run(exercise())
    with pytest.raises(ValueError, match="boolean"):
        create_adapter(AdapterConfig("qq", "qq", {
            "app_id": "102498836", "app_secret_env": "QQ_APP_SECRET", "sandbox": "false"
        }))


def test_qq_business_identifies_corrects_collects_and_accepts_mentioned_guess(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    monkeypatch.delenv("BOT_ADAPTERS_CONFIG", raising=False)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    predictions = []
    store = BotStore(tmp_path)
    predictor = SerializedPredictor(SimpleNamespace(
        device="cpu", threshold=0.65,
        predict_bytes=lambda data: predictions.append(data) or Prediction(
            label="nailong", confidence=0.9, accepted=True,
            scores={"nailong": 0.9, "naiwa": 0.05, "other": 0.05},
        ),
    ))
    calls = []

    async def exercise():
        adapter, recorded = await mock_adapter()
        async def connect_events():
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event(image=True))
            await adapter.handle_event(
                "GROUP_AT_MESSAGE_CREATE",
                event("纠正为奶蛙", id="correction", parent="reply-1"),
            )
            for index, command in enumerate(
                ["奶龙图鉴", "今日含奶量", "今日奶运", "盲盒猜图", "奶蛙", "求赐名"], start=1
            ):
                payload = event(command, id=f"command-{index}")
                if command == "求赐名":
                    payload["author"]["username"] = "王小明"
                await adapter.handle_event(
                    "GROUP_AT_MESSAGE_CREATE", payload
                )
            # Resume 重投同一事件时不能再次执行业务。
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event(image=True))

        monkeypatch.setattr(adapter, "connect", connect_events)
        try:
            await bot.run_adapter(args, adapter, BackendServices(predictor, store))
            calls.extend(recorded)
            sender = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", event()).sender_id
            assert store.milk_family_member(sender) is not None
            assigned = store.milk_name_for_user(sender)
            assert assigned.original_name == "王小明"
            assert "奶" in assigned.milk_name
            assert store.open_guess_round(
                chat_id=adapter.scope.key("chat", "group:group-a")
            ) is None
            ranking = store.guess_ranking(chat_id=adapter.scope.key("chat", "group:group-a"))
            assert ranking
        finally:
            await adapter.close()
    asyncio.run(exercise())
    assert predictions == [image_bytes()]
    observation = store.observation_by_id(1)
    assert observation.asset.corrected
    assert observation.asset.label == "naiwa"
    messages = [data for path, data in calls if path.endswith("/messages")]
    assert any("已纠正" in data.get("content", "") for data in messages)
    assert any("今日含奶量" in data.get("content", "") for data in messages)
    assert any(data["msg_type"] == 7 for data in messages)
    assert all("msg_id" in data and data["msg_seq"] == 1 for data in messages)
    assert all(path.startswith("/v2/groups/group-a/") for path, _ in calls)


def test_qq_quote_cache_group_identity_and_private_reply_context():
    async def exercise():
        adapter, calls = await mock_adapter()
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event(image=True))
            quoted = adapter.normalize_message(
                "GROUP_AT_MESSAGE_CREATE", event(id="quoted", parent="source")
            )
            target = await adapter.resolve_image_target(quoted)
            assert target.file_key == "https://cdn.test/image.png"
            assert await adapter.download_target(target) == image_bytes()
            other = adapter.normalize_message(
                "GROUP_AT_MESSAGE_CREATE", event(group="group-b", parent="source")
            )
            assert await adapter.resolve_image_target(other) != target
            assert (await adapter.resolve_image_target(other)).reason == "quote_unavailable"
            assert other.sender_id != quoted.sender_id
            assert other.message_id != quoted.message_id
            await adapter.handle_event("C2C_MESSAGE_CREATE", event(id="private"))
            private = adapter.normalize_message("C2C_MESSAGE_CREATE", event(id="private"))
            options = {"reply_to": private.message_id}
            result = await adapter.send(private.chat_id, {"text": "奶"}, options)
            assert result.success
            await adapter.send(private.chat_id, {"text": "奶奶"}, options)
            assert calls == [
                ("/v2/users/user-a/messages",
                 {"msg_id": "private", "msg_seq": 1, "msg_type": 0, "content": "奶"}),
                ("/v2/users/user-a/messages",
                 {"msg_id": "private", "msg_seq": 2, "msg_type": 0, "content": "奶奶"}),
            ]
            rejected = await adapter.send(other.chat_id, {"text": "奶"}, options)
            assert not rejected.success
            assert len(calls) == 2
            assert await adapter.resolve(private.sender_id) == "神秘奶友"
            assert await adapter.get_chat_members(other.chat_id) == []
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_qq_new_event_nicknames_mentions_and_reference_indexes_are_scoped():
    async def exercise():
        adapter, _ = await mock_adapter()
        try:
            original = event(image=True)
            original["content"] = ""
            original["author"]["username"] = "王小明"
            original["author"]["id"] = "user-id-alias"
            original["message_scene"] = {
                "ext": ["msg_idx=source-index==", "auth_token=not-for-cache"]
            }
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", original)
            source = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", original)
            assert source.sender_name == "王小明"
            assert await adapter.resolve(source.sender_id) == "王小明"
            markup = adapter.normalize_message(
                "GROUP_AT_MESSAGE_CREATE", event("<@!user-id-alias>", id="markup")
            )
            assert markup.mentions[0].user_id == source.sender_id
            assert markup.mentions[0].name == "王小明"
            quoted = event(id="quote", parent=None)
            quoted["mentions"] = [
                {"member_openid": "member-a", "username": "王小明", "bot": False},
                {"id": "member-b", "username": "小红", "bot": False},
                {"id": "robot", "username": "机器人", "bot": True},
            ]
            quoted["message_scene"] = {"ext": ["ref_msg_idx=source-index=="]}
            message = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", quoted)
            assert message.reply_to_message_id == source.message_id
            assert message.mentions[0].user_id == source.sender_id
            assert message.mentions[1].name == "小红"
            assert len(message.mentions) == 2
            assert (await adapter.resolve_image_target(message)).message_id == source.message_id
            quoted["group_openid"] = "group-b"
            other = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", quoted)
            assert other.mentions[0].user_id != source.sender_id
            assert (await adapter.resolve_image_target(other)).reason == "quote_unavailable"
            assert "not-for-cache" not in repr(adapter._reference_ids)
            original["author"]["bot"] = True
            assert adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", original).sender_is_bot
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_qq_new_quote_event_can_resolve_delivered_image_without_old_message_cache():
    async def exercise():
        adapter, _ = await mock_adapter()
        try:
            data = event(id="new-quote")
            data["message_type"] = 103
            data["message_scene"] = {"ext": ["ref_msg_idx=unseen-image"]}
            data["msg_elements"] = [{
                "msg_elements": [{
                    "attachments": [{"content_type": "image/png", "url": "//cdn.test/quoted.png"}]
                }]
            }]
            message = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", data)
            assert not bot.has_direct_image(message)
            target = await adapter.resolve_image_target(message)
            assert target.reason == "resolved"
            assert target.message_id == message.reply_to_message_id
            assert await adapter.download_target(target) == image_bytes()
            data["msg_elements"] = []
            assert (await adapter.resolve_image_target(adapter.normalize_message(
                "GROUP_AT_MESSAGE_CREATE", data
            ))).reason == "quote_unavailable"
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_qq_uses_shared_image_workflows_and_simple_milk_command(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    monkeypatch.delenv("BOT_ADAPTERS_CONFIG", raising=False)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    # 故意打开旧闪图设置，验证 @发图的识别说明不会被 GIF 发送路径吞掉。
    args.random_push_hdr_probability = 1
    store = BotStore(tmp_path)
    store.register_seed(image_bytes(color="blue"), "nailong")
    store.register_seed(image_bytes(color="green"), "nailong")
    predictor = SerializedPredictor(SimpleNamespace(
        device="cpu", threshold=0.65,
        predict_bytes=lambda data: Prediction(
            label="nailong", confidence=0.9, accepted=True,
            scores={"nailong": 0.9, "naiwa": 0.05, "other": 0.05},
        ),
    ))

    async def exercise():
        adapter, calls = await mock_adapter()

        async def connect_events():
            data = event(image=True)
            data["author"]["username"] = "王小明"
            data["message_scene"] = {"ext": ["msg_idx=image-index"]}
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", data)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert len(messages) == 1
            assert messages[0]["msg_type"] == 7
            assert "奶龙接上了" in messages[0]["content"]
            assert "识别结果：奶龙" in messages[0]["content"]
            assert "奶含量" in messages[0]["content"]
            assert "稀有度" in messages[0]["content"]
            original = store.observation_by_id(1)
            assert original.asset.discoverer_name == "王小明"

            quote = event(id="quote-image")
            quote["message_scene"] = {"ext": ["ref_msg_idx=image-index"]}
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", quote)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert len(messages) == 2
            assert messages[-1]["msg_type"] == 7
            assert "识别结果：奶龙" in messages[-1]["content"]
            assert store.observation_by_id(1).asset.seen_count == 1

            # 无需新增口令：回复原图，沿用飞书“纠正为奶蛙”。
            correction = event("纠正为奶蛙", id="correction-new")
            correction["message_scene"] = {"ext": ["ref_msg_idx=image-index"]}
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", correction)
            assert store.observation_by_id(1).asset.label == "naiwa"

            # 首次发言就能接奶；即使同时点名别人，认领的仍是发送者本人。
            family = event("接奶", id="family")
            family["author"].update(member_openid="member-b", username="小红")
            family["mentions"] = [{"member_openid": "member-a", "username": "王小明"}]
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", family)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 7
            assert "小红，奶家人接奶" in messages[-1]["content"]
            assert "王小明" not in messages[-1]["content"]
            member = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", family)
            assert store.milk_family_member(member.sender_id).display_name == "小红"
            assert len(store.hunter_rankings()) == 1
            assert store.hunter_rankings()[0].display_name == "王小明"

            alias = event("  奶家人  ", id="milk-alias")
            alias["author"]["username"] = "王小明"
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", alias)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert "王小明，奶家人接奶" in messages[-1]["content"]
            before_duplicate = len(messages)
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", alias)
            assert len([payload for path, payload in calls if path.endswith("/messages")]) == (
                before_duplicate
            )

            # 旧的昵称点名不再查询其他人，也不要求手动输入名字或预先打招呼。
            old_mention = event("@王小明", id="old-mention")
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", old_mention)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert "想接奶" in messages[-1]["content"]
            assert "完整昵称" not in messages[-1]["content"]

            private_milk = event("接奶", id="private-milk")
            private_milk["content"] = "接奶"
            await adapter.handle_event("C2C_MESSAGE_CREATE", private_milk)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 7
            assert "奶家人接奶" in messages[-1]["content"]
            assert calls[-1][0] == "/v2/users/user-a/messages"

            help_request = event("帮助", id="qq-help")
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", help_request)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert "回复图片并 @我" in messages[-1]["content"]
            assert "当前未开启定时主动播报" in messages[-1]["content"]
            assert "接龙 + 图片" not in messages[-1]["content"]
            assert "接奶（或奶家人）" in messages[-1]["content"]
            assert "手动输入" not in messages[-1]["content"]

            # 真实群事件把图片放在引用子节点里，顶层 content/attachments 都为空。
            native_quote = event(id="native-quote")
            native_quote["content"] = ""
            native_quote["message_type"] = 103
            native_quote["parallel_message"] = False
            native_quote["message_scene"] = {"ext": ["ref_msg_idx=TMP-uncached"]}
            native_quote["msg_elements"] = [{
                "msg_idx": "TMP-uncached", "message_type": 103, "content": "图片",
                "attachments": [{
                    "content_type": "image/jpeg", "url": "//cdn.test/image.jpg"
                }],
            }]
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", native_quote)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 7
            assert "奶蛙接上了" in messages[-1]["content"]

            # 群主开全量消息后，不 @的图片仍走和飞书一致的自动接龙。
            full = event(id="full-image", image=True)
            full["content"] = ""
            await adapter.handle_event("GROUP_MESSAGE_CREATE", full)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 7
            assert store.observation_by_id(1).asset.seen_count == 3

            private = event(id="private-image", image=True)
            private["author"]["username"] = "小红"
            await adapter.handle_event("C2C_MESSAGE_CREATE", private)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 0
            assert "奶龙接上了" not in messages[-1]["content"]
            assert calls[-1][0] == "/v2/users/user-a/messages"

            # 只 @机器人时提示新口令，不再要求输入群友昵称。
            bare = event(id="qq-stripped-mention")
            bare["content"] = ""
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", bare)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert "想接奶" in messages[-1]["content"]
            assert "手动输入" not in messages[-1]["content"]

            # 图片与引用仍按原逻辑识图、接龙和纠正。
            mixed = event("@王小明", id="image-and-name", image=True)
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", mixed)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 7
            assert "已经确认是奶蛙" in messages[-1]["content"]
            assert "奶含量" in messages[-1]["content"]
            assert "同名" not in messages[-1]["content"]
            mixed_quote = event("@王小明", id="quote-and-name", parent="source")
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", mixed_quote)
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert messages[-1]["msg_type"] == 7
            assert "已经确认是奶蛙" in messages[-1]["content"]
            assert "稀有度" in messages[-1]["content"]

        monkeypatch.setattr(adapter, "connect", connect_events)
        try:
            await bot.run_adapter(args, adapter, BackendServices(predictor, store))
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_qq_take_milk_handles_empty_catalogue_and_requires_group_activation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    monkeypatch.delenv("BOT_ADAPTERS_CONFIG", raising=False)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    store = BotStore(tmp_path)
    predictor = SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.65))

    async def exercise():
        adapter, calls = await mock_adapter()

        async def connect_events():
            ordinary = event("接奶", id="unmentioned-milk")
            ordinary["content"] = "接奶"
            await adapter.handle_event("GROUP_MESSAGE_CREATE", ordinary)
            assert not calls

            first = event("接奶", id="first-milk")
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", first)
            sender = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", first)
            assert store.milk_family_member(sender.sender_id) is not None
            messages = [payload for path, payload in calls if path.endswith("/messages")]
            assert len(messages) == 1
            assert messages[0]["msg_type"] == 0
            assert "你已经是奶家人啦" in messages[0]["content"]
            assert "先发张奶龙或奶蛙图片" in messages[0]["content"]
            assert "图片消息后 @我" not in messages[0]["content"]

        monkeypatch.setattr(adapter, "connect", connect_events)
        try:
            await bot.run_adapter(args, adapter, BackendServices(predictor, store))
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_token_cache_refresh_is_bounded_and_shared_by_concurrent_requests():
    async def exercise():
        tokens = []
        requests = []

        def auth(request):
            token = f"token-{len(tokens)}"
            tokens.append(token)
            return httpx.Response(200, json={"access_token": token, "expires_in": 7200})

        def api(request):
            token = request.headers["Authorization"]
            requests.append(token)
            if token == "QQBot token-0":
                return httpx.Response(401, json={"code": 11244})
            return httpx.Response(200, json={"url": "wss://gateway.test"})

        adapter, _ = await mock_adapter(api=api, auth=auth)
        try:
            assert len(await asyncio.gather(*(
                adapter._call("GET", "/gateway") for _ in range(4)
            ))) == 4
            assert tokens == ["token-0", "token-1"]
            assert requests.count("QQBot token-0") == 1
            adapter._expires_at = datetime.min.replace(tzinfo=UTC)
            await adapter._call("GET", "/gateway")
            assert tokens == ["token-0", "token-1", "token-2"]
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_gallery_upload_is_one_png_and_reply_uses_file_info():
    async def exercise():
        adapter, calls = await mock_adapter()
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            chat_id = adapter.scope.key("chat", "group:group-a")
            result = await adapter.send(chat_id, {
                "gallery": [
                    {"source": image_bytes("GIF"), "caption": f"#{id:04d} · 奶龙"}
                    for id in range(1, 10)
                ],
                "caption": "奶图图鉴",
            })
            assert result.success
            assert len(calls) == 2
            upload = calls[0][1]
            assert upload["file_type"] == 1
            assert upload["srv_send_msg"] is False
            data = base64.b64decode(upload["file_data"])
            with Image.open(io.BytesIO(data)) as image:
                assert image.format == "PNG"
                assert image.size == (960, 750)
            assert calls[1][1]["media"] == {"file_info": "test-file"}
            parent = json.loads(adapter.scope.raw("message", result.message_id))[1]
            quoted = adapter.normalize_message(
                "GROUP_AT_MESSAGE_CREATE", event(id="quote-gallery", parent=parent)
            )
            target = await adapter.resolve_image_target(quoted)
            assert await adapter.download_target(target) == data
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_proactive_requires_opt_in_and_expired_explicit_replies_are_rejected():
    async def exercise():
        adapter, calls = await mock_adapter()
        chat_id = adapter.scope.key("chat", "group:group-a")
        try:
            result = await adapter.send(chat_id, {"text": "日报"})
            assert result.error.code == ErrorCode.NOT_SUPPORTED
            assert calls == []
            adapter.allow_proactive = True
            assert (await adapter.send(chat_id, {"text": "日报"})).success
            assert "msg_id" not in calls[0][1]
            private = adapter.scope.key("chat", "p2p:user-a")
            private_result = await adapter.send(private, {"text": "日报"})
            assert private_result.error.code == ErrorCode.NOT_SUPPORTED
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            message = adapter.normalize_message("GROUP_AT_MESSAGE_CREATE", event())
            adapter._received_at[message.message_id] = (
                datetime.now(UTC) - timedelta(minutes=6)
            )
            expired = await adapter.send(
                chat_id, {"text": "奶"}, {"reply_to": message.message_id}
            )
            assert not expired.success
            assert len(calls) == 1
        finally:
            await adapter.close()
    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("upload", "status", "expected"),
    [(False, 403, ErrorCode.UNKNOWN), (True, 400, ErrorCode.UPLOAD_FAILED)],
)
def test_send_errors_do_not_deactivate_chat_or_retry_unknown_results(upload, status, expected):
    async def exercise():
        calls = []

        def api(request):
            calls.append(request.url.path)
            return httpx.Response(status, json={"code": 304023})

        adapter, _ = await mock_adapter(api=api)
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            content = {"image": {"source": image_bytes()}} if upload else {"text": "奶"}
            result = await adapter.send(adapter.scope.key("chat", "group:group-a"), content)
            assert result.error.code == expected
            assert len(calls) == 1
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_uncertain_send_timeout_is_not_retried():
    async def exercise():
        calls = []

        def api(request):
            calls.append(request.url.path)
            raise httpx.ReadTimeout("mock timeout", request=request)

        adapter, _ = await mock_adapter(api=api)
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            result = await adapter.send(
                adapter.scope.key("chat", "group:group-a"), {"text": "奶"}
            )
            assert result.error.code == ErrorCode.SEND_TIMEOUT
            assert calls == ["/v2/groups/group-a/messages"]
        finally:
            await adapter.close()
    asyncio.run(exercise())


def test_websocket_missing_heartbeat_ack_closes_session(monkeypatch):
    async def exercise():
        closed = asyncio.Event()
        failures = []

        def api(request):
            return httpx.Response(200, json={"url": "wss://gateway.test/"})

        adapter, _ = await mock_adapter(api=api)

        async def gateway(socket):
            try:
                await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 20}}))
                assert json.loads(await socket.recv())["op"] == 2
                await socket.send(json.dumps({
                    "op": 0, "t": "READY", "s": 1, "d": {"session_id": "no-ack"}
                }))
                assert json.loads(await socket.recv()) == {"op": 1, "d": 1}
                # 故意不返回 ACK，客户端应关闭而非维持一个假在线连接。
                await socket.wait_closed()
                assert socket.close_code == 4000
            except Exception as error:
                failures.append(error)
            finally:
                closed.set()

        async with serve(gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            monkeypatch.setattr(
                "nailong_classifier.adapters.qq.connect",
                lambda url, **kwargs: ws_connect(f"ws://127.0.0.1:{port}", **kwargs),
            )
            task = asyncio.create_task(adapter.connect())
            try:
                await asyncio.wait_for(closed.wait(), timeout=3)
                assert not failures
            finally:
                await adapter.close()
                await asyncio.wait_for(task, timeout=2)
    asyncio.run(exercise())


def test_websocket_identify_heartbeat_resume_and_invalid_session(monkeypatch):
    async def exercise():
        connections = []
        first_message_started = asyncio.Event()
        release_handler = asyncio.Event()
        finished = asyncio.Event()
        failures = []
        ids = itertools.count(1)
        received = []

        def api(request):
            assert request.headers["Authorization"] == "QQBot test-token"
            return httpx.Response(200, json={"url": "wss://gateway.test/"})

        adapter, _ = await mock_adapter(api=api)

        async def handler(message):
            received.append(message.message_id)
            first_message_started.set()
            await release_handler.wait()

        adapter.on("message", handler)

        async def gateway(socket):
            try:
                index = next(ids)
                await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 20}}))
                identify = json.loads(await socket.recv())
                connections.append(identify)
                if index == 1:
                    assert identify == {
                        "op": 2, "d": {
                            "token": "QQBot test-token", "intents": 33_554_432, "shard": [0, 1]
                        }
                    }
                    await socket.send(json.dumps({
                        "op": 0, "t": "READY", "s": 1, "d": {"session_id": "session-a"}
                    }))
                    await socket.send(json.dumps({
                        "op": 0, "t": "GROUP_AT_MESSAGE_CREATE", "s": 2, "d": event()
                    }))
                    await first_message_started.wait()
                    heartbeat = json.loads(await socket.recv())
                    assert heartbeat == {"op": 1, "d": 2}
                    # 业务还在等待，但 ACK 能正常处理。
                    await socket.send(json.dumps({"op": 11}))
                    await socket.send(json.dumps({"op": 7}))
                    release_handler.set()
                elif index == 2:
                    assert identify == {
                        "op": 6, "d": {
                            "token": "QQBot test-token", "session_id": "session-a", "seq": 2
                        }
                    }
                    await socket.send(json.dumps({"op": 9, "d": False}))
                else:
                    assert identify["op"] == 2
                    finished.set()
                    await socket.wait_closed()
            except Exception as error:
                failures.append(error)
                finished.set()

        async with serve(gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            monkeypatch.setattr(
                "nailong_classifier.adapters.qq.connect",
                lambda url, **kwargs: ws_connect(f"ws://127.0.0.1:{port}", **kwargs),
            )
            task = asyncio.create_task(adapter.connect())
            try:
                await asyncio.wait_for(finished.wait(), timeout=10)
                assert not failures
                assert len(connections) == 3
                assert len(received) == 1
            finally:
                release_handler.set()
                await adapter.close()
                await asyncio.wait_for(task, timeout=2)
    asyncio.run(exercise())
