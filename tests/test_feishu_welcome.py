from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

import nailong_classifier.adapters.feishu as feishu
import nailong_classifier.bot as bot
from nailong_classifier.adapters.base import IdentityScope, SendError
from nailong_classifier.adapters.feishu_process import ProcessFeishuAdapter
from nailong_classifier.services import BackendServices, SerializedPredictor
from nailong_classifier.storage import BotStore

EVENT_TYPE = "im.chat.member.user.added_v1"


def run_welcomes(
    tmp_path, monkeypatch, events, *, legacy=False, lookup="李雷",
    send_outcomes=(True,), concurrent=False,
):
    """假 SDK 注入原始事件，走真实归一化、父进程 ACK 与共享欢迎业务。"""
    sent, lookups, acknowledgements = [], [], []
    scope = IdentityScope("feishu", "welcome-test", legacy=legacy)
    store = BotStore(tmp_path)
    monkeypatch.setattr(
        bot, "milk_name_options",
        lambda name: iter((f"奶蛙最爱的 {bot.milk_nickname_variants(name)[0]}",)),
    )

    class Channel:
        client = SimpleNamespace()

        def __init__(self, **kwargs):
            self.raw_handlers = {}

        def on(self, event, handler):
            pass

        def on_raw_event(self, event, handler):
            self.raw_handlers[event] = handler

        async def connect(self):
            callback = self.raw_handlers[EVENT_TYPE]
            if concurrent:
                await asyncio.gather(*(callback(event) for event in events))
            else:
                for event in events:
                    await callback(event)

        async def send(self, chat_id, content, options=None):
            sent.append((chat_id, content))
            outcome = send_outcomes[min(len(sent) - 1, len(send_outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(
                success=outcome, message_id="welcome-response" if outcome else None,
                error=None if outcome else SendError("test_failure", "private error details"),
            )

        async def get_chat_members(self, chat_id, **kwargs):
            return []

    monkeypatch.setattr(feishu, "FeishuChannel", Channel)
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    args.douyin_cookie_file = None
    services = BackendServices(
        SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.65)), store
    )

    async def exercise():
        sdk = feishu.FeishuAdapter(scope, app_id="dummy", app_secret="dummy")
        parent = ProcessFeishuAdapter(scope, app_id="dummy", app_secret="dummy")

        async def resolve(raw_id, *, fallback):
            lookups.append(raw_id)
            if isinstance(lookup, Exception):
                raise lookup
            return lookup or fallback

        async def acknowledge(frame):
            acknowledgements.append(frame)

        async def rpc(method, *args, **kwargs):
            return await getattr(sdk, method)(*args, **kwargs)

        async def forward(payload):
            await parent._dispatch("test-event", "members_added", payload)

        sdk.contacts.resolve = resolve
        sdk.on("members_added", forward)
        parent._rpc = rpc
        parent._send = acknowledge
        await bot.run_adapter(args, parent, services)
        # 所有平台调用都已替换为本地假 SDK，不启动真实 worker 或访问网络。
        assert parent._process is None

    asyncio.run(exercise())
    return store, scope, sent, lookups, acknowledgements


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("envelope", [False, True])
def test_member_events_welcome_nested_ids_missing_names_and_deduplicate(
    tmp_path, monkeypatch, caplog, legacy, envelope,
):
    caplog.set_level(logging.INFO)
    data = {
        "chat_id": "private-chat",
        "users": [
            {"user_id": {"open_id": "private-a", "user_id": "different-id"}, "name": "王小明"},
            {"user_id": {"open_id": "private-b"}},
            {"open_id": "private-c", "name": " 张雪奶 "},
            {"user_id": "private-d", "name": "赵小李"},
            {"user_id": {"open_id": "private-a"}, "name": "重复昵称"},
            {"user_id": {"union_id": "wrong-id"}, "name": "无效"},
            None,
        ],
    }
    event = {"event": data} if envelope else data
    store, scope, sent, lookups, acks = run_welcomes(
        tmp_path, monkeypatch, [event, event], legacy=legacy,
    )
    assert lookups == ["private-b"]
    assert len(sent) == 1
    assert sent[0][0] == "private-chat"
    text = sent[0][1]["text"]
    assert "王小明→奶蛙最爱的 奶小明" in text
    assert "李雷→奶蛙最爱的 奶雷" in text
    assert "张雪奶→奶蛙最爱的 张雪奶" in text
    assert "赵小李→奶蛙最爱的 奶小李" in text
    assert "重复昵称" not in text
    for raw_id, name in [("private-a", "王小明"), ("private-b", "李雷")]:
        assert store.milk_family_member(scope.key("user", raw_id)).display_name == name
        assert store.milk_name_for_user(scope.key("user", raw_id)).milk_name == (
            f"奶蛙最爱的 {bot.milk_nickname_variants(name)[0]}"
        )
    assert len(acks) == 2
    assert all(frame == ("ack", "test-event") for frame in acks)
    assert "Feishu member event received" in caplog.text
    assert "members=4 missing_names=1" in caplog.text
    assert "Milk name welcome sent" in caplog.text
    assert "reason=no_new_members" in caplog.text
    assert all(value not in caplog.text for value in ("private-chat", "private-a", "王小明"))

    # 服务重启后成功记录仍然生效，不能给同一个人再发一次。
    assert not BotStore(tmp_path).claim_milk_name_welcome(
        scope.key("chat", "private-chat"), scope.key("user", "private-a")
    )


@pytest.mark.parametrize("lookup", ["", RuntimeError("private lookup error"), "   "])
def test_missing_contact_permissions_still_welcome_newcomer(tmp_path, monkeypatch, lookup):
    event = {"event": {"chat_id": "chat", "users": [{"user_id": {"open_id": "user"}}]}}
    store, scope, sent, _, _ = run_welcomes(tmp_path, monkeypatch, [event], lookup=lookup)
    assert sent[0][1]["text"] == "新奶友「新朋友」进群，奶名已定：奶蛙最爱的 奶朋友。"
    assert store.milk_family_member(scope.key("user", "user")).display_name == "新朋友"


@pytest.mark.parametrize("failure", [False, RuntimeError("private send error")])
def test_failed_batch_welcome_can_retry_without_rewelcoming_successes(
    tmp_path, monkeypatch, caplog, failure,
):
    caplog.set_level(logging.INFO)

    def newcomer(user, name):
        return {"user_id": {"open_id": user}, "name": name}

    first = {"event": {"chat_id": "chat", "users": [newcomer("old", "王小明")]}}
    batch = {"event": {"chat_id": "chat", "users": [
        newcomer("old", "王小明"), newcomer("new-a", "李雷"), newcomer("new-b", "赵小李"),
    ]}}
    _, _, sent, _, acks = run_welcomes(
        tmp_path, monkeypatch, [first, batch, batch, batch],
        send_outcomes=(True, failure, True),
    )
    assert len(sent) == 3
    assert "王小明" in sent[0][1]["text"]
    assert "王小明" not in sent[1][1]["text"]
    assert sent[1][1] == sent[2][1]
    assert "李雷→奶蛙最爱的 奶雷" in sent[2][1]["text"]
    assert "赵小李→奶蛙最爱的 奶小李" in sent[2][1]["text"]
    assert len(acks) == 4
    assert "private send error" not in caplog.text
    assert "private error details" not in caplog.text


def test_concurrent_duplicate_member_events_send_one_welcome(tmp_path, monkeypatch):
    event = {"event": {"chat_id": "chat", "users": [{"open_id": "user", "name": "王小明"}]}}
    _, _, sent, _, acks = run_welcomes(
        tmp_path, monkeypatch, [event, event, event], concurrent=True,
    )
    assert len(sent) == 1
    assert len(acks) == 3


def test_newcomer_names_use_random_factory_and_same_nicknames_still_stay_unique(
    tmp_path, monkeypatch,
):
    event = {"event": {"chat_id": "chat", "users": [
        {"open_id": "first", "name": "韩婉清"},
        {"open_id": "second", "name": "韩婉清"},
    ]}}
    store, scope, sent, _, _ = run_welcomes(
        tmp_path, monkeypatch, [event, event],
    )
    first = store.milk_name_for_user(scope.key("user", "first"))
    second = store.milk_name_for_user(scope.key("user", "second"))
    assert len(sent) == 1
    assert "奶蛙最爱的 奶婉清" in {first.milk_name, second.milk_name}
    assert first.milk_name != second.milk_name
    assert first.original_name == second.original_name == "韩婉清"


@pytest.mark.parametrize("event", [
    None,
    {"event": None},
    {"event": {"chat_id": "", "users": []}},
    {"event": {"chat_id": "chat", "users": {}}},
    {"event": {"chat_id": "chat", "users": [None, {"name": "无身份"}]}},
    {"event": {"chat_id": "chat", "users": []}},
])
def test_malformed_or_empty_member_events_are_diagnosed_without_sending(
    tmp_path, monkeypatch, caplog, event,
):
    caplog.set_level(logging.INFO)
    _, _, sent, lookups, _ = run_welcomes(tmp_path, monkeypatch, [event])
    assert not sent
    assert not lookups
    assert "Feishu member event received" in caplog.text
