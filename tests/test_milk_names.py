from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from types import SimpleNamespace

import pytest

import nailong_classifier.bot as bot
from nailong_classifier.adapters.base import (
    IdentityScope,
    Member,
    MembersAdded,
    Mention,
    Message,
    SendError,
    SendResult,
)
from nailong_classifier.services import BackendServices, SerializedPredictor
from nailong_classifier.storage import BotStore


def test_milk_name_survives_restart_and_nickname_changes_without_storing_raw_identity(tmp_path):
    store = BotStore(tmp_path)
    assigned = store.assign_milk_name(
        "private-user", original_name="王小明", milk_name="王小奶"
    )
    store.record_milk_family_member("private-user", display_name="李雷")
    restarted = BotStore(tmp_path)
    assert restarted.milk_name_for_user("private-user") == assigned
    assert restarted.assign_milk_name(
        "private-user", original_name="李雷", milk_name="李奶"
    ) == assigned
    assert restarted.milk_family_member("private-user").display_name == "李雷"
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT user_hash, original_name, milk_name FROM milk_names"
        ).fetchall() == [(store.identity_hash("private-user"), "王小明", "王小奶")]


def test_simultaneous_welcomes_keep_one_mapping_and_disambiguate_same_nicknames(tmp_path):
    store = BotStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda name: store.assign_milk_name(
                "user-a", original_name=name, milk_name=bot.make_milk_name(name)
            ),
            ["王小明", "李雷", "张小强", "赵小李"],
        ))
    assert all(result == results[0] for result in results)
    second = store.assign_milk_name(
        "user-b", original_name=results[0].original_name, milk_name=results[0].milk_name
    )
    assert second.milk_name != results[0].milk_name
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM milk_names").fetchone()[0] == 2


def exercise_names(
    tmp_path, monkeypatch, *, platform="feishu", chat_type="group", mentioned=True,
    nickname="王小明", lookup="李雷", welcome=False, send_outcomes=(True,),
    friend=False, quiet=False, real_words=False,
):
    scope = IdentityScope(platform, "name-test")
    # 路由与持久化测试固定名字主体；随机位置由独立生成器测试覆盖。
    if not real_words:
        monkeypatch.setattr(
            bot, "make_milk_name",
            lambda name: {
                "王小明": "王小奶", "李雷": "李奶",
                "张雪奶": "张雪奶", "神秘奶友": "神秘奶友",
            }[name],
        )
        monkeypatch.setattr(
            bot, "milk_name_options",
            lambda name: (
                f"奶蛙最爱的 {bot.make_milk_name(name)}",
                f"最奶的 {bot.make_milk_name(name)}",
            ),
        )
    store = BotStore(tmp_path)
    sent, lookups = [], []
    user_id = scope.key("user", "sender")
    if quiet:
        store.record_chat(scope.key("chat", "request-chat"), "group")
        store.set_proactive_enabled(scope.key("chat", "request-chat"), False)

    class Adapter:
        def __init__(self):
            self.scope = scope
            self.handlers = {}

        def on(self, event, handler):
            self.handlers[event] = handler

        async def resolve(self, user_id, *, fallback):
            lookups.append(user_id)
            if isinstance(lookup, Exception):
                raise lookup
            return lookup or fallback

        async def get_chat_members(self, chat_id):
            return []

        async def send(self, chat_id, content, options=None):
            sent.append((chat_id, content, options))
            outcome = send_outcomes[min(len(sent) - 1, len(send_outcomes) - 1)]
            return SendResult(
                outcome, "reply" if outcome else None,
                None if outcome else SendError("test_failure"),
            )

        async def connect(self):
            if welcome:
                await self.handlers["members_added"](MembersAdded(
                    scope.key("chat", "welcome-chat"),
                    (Member(user_id, "王小明"),),
                ))
            for index in range(2):
                current_nickname = nickname[index] if isinstance(nickname, tuple) else nickname
                await self.handlers["message"](Message(
                    message_id=scope.key("message", f"request-{index}"),
                    chat_id=scope.key("chat", "request-chat"),
                    chat_type=chat_type,
                    sender_id=user_id,
                    sender_name=current_nickname,
                    mentioned_bot=mentioned,
                    body_text="求赐名" if index == 0 else " 求 赐名 ",
                    mentions=(Mention(scope.key("user", "friend"), "李雷"),) if friend else (),
                ))

    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    monkeypatch.delenv("BOT_ADAPTERS_CONFIG", raising=False)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.douyin_cookie_file = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    services = BackendServices(
        SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.5)), store
    )
    asyncio.run(bot.run_adapter(args, Adapter(), services))
    return store, user_id, sent, lookups


@pytest.mark.parametrize("platform", ["feishu", "qq"])
@pytest.mark.parametrize("nickname,expected", [("王小明", "王小奶"), ("张雪奶", "张雪奶")])
def test_grant_command_records_sender_and_changes_name_on_each_request(
    tmp_path, monkeypatch, platform, nickname, expected,
):
    store, user_id, sent, lookups = exercise_names(
        tmp_path, monkeypatch, platform=platform, nickname=nickname
    )
    assert len(sent) == 2
    assert sent[0][1] != sent[1][1]
    assert f"奶蛙最爱的 {expected}" in sent[0][1]["text"]
    assert f"最奶的 {expected}" in sent[1][1]["text"]
    assert IdentityScope(platform, "name-test").raw(
        "message", sent[0][2]["reply_to"]
    ) == "request-0"
    assert store.milk_name_for_user(user_id).milk_name == f"最奶的 {expected}"
    assert store.milk_family_member(user_id) is not None
    assert not lookups


@pytest.mark.parametrize("platform", ["feishu", "qq"])
def test_private_grant_does_not_require_mention(tmp_path, monkeypatch, platform):
    store, user_id, sent, _ = exercise_names(
        tmp_path, monkeypatch, platform=platform, chat_type="p2p", mentioned=False
    )
    assert len(sent) == 2
    assert store.milk_name_for_user(user_id).milk_name == "最奶的 王小奶"


def test_group_without_mention_does_not_grant_name(tmp_path, monkeypatch):
    store, user_id, sent, lookups = exercise_names(tmp_path, monkeypatch, mentioned=False)
    assert not sent
    assert not lookups
    assert store.milk_name_for_user(user_id) is None


def test_grant_for_mentioned_member_keeps_existing_name_and_public_lookup_requests_image(
    tmp_path, monkeypatch,
):
    scope = IdentityScope("feishu", "name-test")
    store = BotStore(tmp_path)
    sent = []
    sender_id = scope.key("user", "sender")
    friend_id = scope.key("user", "friend")
    monkeypatch.setattr(bot, "make_milk_name", lambda name: f"{name}奶")
    monkeypatch.setattr(
        bot,
        "milk_name_options",
        lambda name: iter((f"奶龙的一号{name}奶", f"奶龙的二号{name}奶")),
    )

    class Adapter:
        def __init__(self):
            self.scope = scope
            self.handlers = {}

        def on(self, event, handler):
            self.handlers[event] = handler

        async def resolve(self, user_id, *, fallback):
            return fallback

        async def get_chat_members(self, chat_id):
            return []

        async def send(self, chat_id, content, options=None):
            sent.append((content, options))
            return SendResult(True, "reply")

        async def connect(self):
            await self.handlers["message"](Message(
                message_id=scope.key("message", "image-bot-introduction"),
                chat_id=scope.key("chat", "group"),
                chat_type="group",
                sender_id=scope.key("user", "ou_image_bot"),
                sender_name="神笔奶良",
                sender_is_bot=True,
            ))
            for index, body_text in enumerate(("给奶友求赐名", "给奶友求赐名", "奶友")):
                await self.handlers["message"](Message(
                    message_id=scope.key("message", f"message-{index}"),
                    chat_id=scope.key("chat", "group"),
                    chat_type="group",
                    sender_id=sender_id,
                    sender_name="发起人",
                    mentioned_bot=True,
                    body_text=body_text,
                    mentions=(Mention(friend_id, "奶友"),),
                ))

    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    monkeypatch.delenv("BOT_ADAPTERS_CONFIG", raising=False)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.douyin_cookie_file = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    services = BackendServices(
        SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.5)), store
    )
    asyncio.run(bot.run_adapter(args, Adapter(), services))

    assigned = store.milk_name_for_user(friend_id)
    assert assigned is not None
    assert assigned.milk_name == "奶龙的二号奶友奶"
    assert "已入奶籍，奶名是【奶龙的一号奶友奶】" in sent[0][0]["text"]
    assert "奶友 是奶家人，是你【奶龙的二号奶友奶】" in sent[2][0]["text"]
    assert "奶友 是奶家人，是你【奶龙的二号奶友奶】" in sent[4][0]["text"]
    assert sent[1][0]["post"] == bot.build_milk_name_image_request(
        "ou_image_bot", "奶龙的一号奶友奶"
    )
    assert sent[3][0]["post"] == sent[5][0]["post"] == bot.build_milk_name_image_request(
        "ou_image_bot", "奶龙的二号奶友奶"
    )


@pytest.mark.parametrize("lookup,expected", [
    ("李雷", "李奶"),
    (RuntimeError("private lookup failure"), "神秘奶友"),
])
def test_missing_nickname_looks_up_contact_once_and_has_permission_fallback(
    tmp_path, monkeypatch, caplog, lookup, expected,
):
    store, user_id, sent, lookups = exercise_names(
        tmp_path, monkeypatch, nickname=None, lookup=lookup
    )
    assert len(lookups) == (2 if isinstance(lookup, Exception) else 1)
    if not isinstance(lookup, Exception):
        assert expected in sent[0][1]["text"]
    else:
        assert "奶名已经记住啦" in sent[0][1]["text"]
    assert store.milk_name_for_user(user_id).milk_name == f"最奶的 {expected}"
    assert "private lookup failure" not in caplog.text


def test_later_message_nickname_repairs_fallback_original_before_renaming(tmp_path, monkeypatch):
    store, user_id, sent, lookups = exercise_names(
        tmp_path, monkeypatch, nickname=(None, "李雷"), lookup=None,
    )
    assert len(lookups) == 1
    assert "奶名已经记住啦" in sent[0][1]["text"]
    assert "李雷 → 奶蛙最爱的 李奶" in sent[1][1]["text"]
    assigned = store.milk_name_for_user(user_id)
    assert assigned.original_name == "李雷"
    assert assigned.original_name_known


@pytest.mark.parametrize("source", ["message", "mentions", "welcome"])
def test_real_nickname_fills_placeholder_without_changing_milk_name(tmp_path, source):
    store = BotStore(tmp_path)
    original = store.assign_milk_name(
        "user", original_name="神秘奶友", milk_name="尼古拉斯·奶",
        original_name_known=False,
    )
    if source == "message":
        store.record_milk_family_member("user", display_name="李雷")
    elif source == "mentions":
        store.record_milk_family_members([("user", "李雷")])
    else:
        store.assign_milk_name("user", original_name="李雷", milk_name="李奶")
    updated = store.milk_name_for_user("user")
    assert updated.original_name == "李雷"
    assert updated.original_name_known
    assert updated.milk_name == original.milk_name
    assert updated.assigned_at == original.assigned_at
    store.record_milk_family_member("user", display_name="王小明")
    assert store.milk_name_for_user("user") == updated


def test_message_mention_repairs_another_users_saved_fallback_name(tmp_path, monkeypatch):
    friend_id = IdentityScope("feishu", "name-test").key("user", "friend")
    prior = BotStore(tmp_path).assign_milk_name(
        friend_id, original_name="神秘奶友", milk_name="尼古拉斯·奶",
        original_name_known=False,
    )
    store, _, _, _ = exercise_names(tmp_path, monkeypatch, friend=True)
    repaired = store.milk_name_for_user(friend_id)
    assert repaired.original_name == "李雷"
    assert repaired.original_name_known
    assert repaired.milk_name == prior.milk_name
    assert repaired.assigned_at == prior.assigned_at


def test_literal_fallback_nickname_is_not_treated_as_unknown_for_new_records(tmp_path):
    store = BotStore(tmp_path)
    assigned = store.assign_milk_name(
        "user", original_name="神秘奶友", milk_name="尼古拉斯·奶",
        original_name_known=True,
    )
    store.record_milk_family_member("user", display_name="李雷")
    assert BotStore(tmp_path).milk_name_for_user("user") == assigned


def test_legacy_fallback_migration_uses_already_known_member_name(tmp_path):
    store = BotStore(tmp_path)
    assigned = store.assign_milk_name(
        "user", original_name="神秘奶友", milk_name="尼古拉斯·奶",
    )
    store.record_milk_family_member("user", display_name="李雷")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("ALTER TABLE milk_names DROP COLUMN original_name_known")
    updated = BotStore(tmp_path).milk_name_for_user("user")
    assert updated.original_name == "李雷"
    assert updated.original_name_known
    assert updated.milk_name == assigned.milk_name
    assert updated.assigned_at == assigned.assigned_at


def test_welcome_and_active_grant_share_one_mapping_across_groups(tmp_path, monkeypatch):
    store, user_id, sent, _ = exercise_names(
        tmp_path, monkeypatch, welcome=True, nickname="李雷"
    )
    assert len(sent) == 3
    assert "王小奶" in sent[0][1]["text"]
    assert "王小奶" in sent[1][1]["text"]
    assert "李奶" not in sent[1][1]["text"]
    assert store.milk_name_for_user(user_id).original_name == "王小明"
    assert "最奶的 王小奶" in sent[1][1]["text"]
    assert store.milk_name_for_user(user_id).milk_name == "奶蛙最爱的 王小奶"
    assert store.milk_family_member(user_id).display_name == "李雷"


def test_failed_reply_does_not_prevent_next_explicit_request_from_renaming(tmp_path, monkeypatch):
    store, user_id, sent, _ = exercise_names(
        tmp_path, monkeypatch, send_outcomes=(False, True)
    )
    assert sent[0][1] != sent[1][1]
    assert store.milk_name_for_user(user_id).milk_name == "最奶的 王小奶"


def test_grant_names_sender_instead_of_another_mentioned_member(tmp_path, monkeypatch):
    store, user_id, sent, _ = exercise_names(tmp_path, monkeypatch, friend=True)
    assert "王小奶" in sent[0][1]["text"]
    assert store.milk_name_for_user(user_id).milk_name == "最奶的 王小奶"
    assert store.milk_name_for_user(
        IdentityScope("feishu", "name-test").key("user", "friend")
    ) is None


def test_explicit_grant_still_works_when_proactive_messages_are_disabled(tmp_path, monkeypatch):
    store, user_id, sent, _ = exercise_names(tmp_path, monkeypatch, quiet=True)
    assert len(sent) == 2
    assert store.milk_name_for_user(user_id).milk_name == "最奶的 王小奶"


@pytest.mark.parametrize("style,expected", [
    ("nickname", "奶蛙最爱的 奶小明"),
    ("foreign", "尼古拉斯·奶龙·奶小明"),
    ("foreign_milk", "尼古拉斯·奶龙大侠"),
    ("title", "奶龙大侠"),
    ("title_name", "奶龙大侠·奶小明"),
])
def test_word_generator_combines_each_style(monkeypatch, style, expected):
    monkeypatch.setattr(
        bot.secrets, "choice",
        lambda options: style if "nickname" in options else options[0],
    )
    assert set(bot.milk_name_options("王小明")) == {expected}


@pytest.mark.parametrize("style", ["nickname", "foreign", "foreign_milk", "title", "title_name"])
@pytest.mark.parametrize("original", ["韩婉清", "张雪奶", "神秘奶友"])
def test_all_name_styles_keep_explicit_character_connection(monkeypatch, style, original):
    real_choice = bot.secrets.choice
    monkeypatch.setattr(
        bot.secrets, "choice",
        lambda options: style if "nickname" in options else real_choice(options),
    )
    candidates = list(bot.milk_name_options(original))
    assert candidates
    assert all("奶龙" in name or "奶蛙" in name for name in candidates)
    if style in {"nickname", "foreign", "title_name"}:
        assert all(any(
            variant in name for variant in bot.milk_nickname_variants(original)
        ) for name in candidates)


@pytest.mark.parametrize("platform", ["feishu", "qq"])
def test_real_word_bank_works_for_welcome_and_repeated_grant(tmp_path, monkeypatch, platform):
    store, user_id, sent, _ = exercise_names(
        tmp_path, monkeypatch, platform=platform, welcome=True, real_words=True,
    )
    assert len(sent) == 3
    assert all("奶龙" in content["text"] or "奶蛙" in content["text"] for _, content, _ in sent)
    assert sent[1][1]["text"] != sent[2][1]["text"]
    assigned = store.milk_name_for_user(user_id)
    assert assigned.milk_name in sent[2][1]["text"]
    assert assigned.original_name == "王小明"
    assert BotStore(tmp_path).milk_name_for_user(user_id) == assigned


def test_renewal_survives_restart_without_resetting_original_or_welcome(tmp_path):
    store = BotStore(tmp_path)
    initial = store.assign_milk_name("user", original_name="王小明", milk_name="王小奶")
    assert store.claim_milk_name_welcome("chat", "user")
    renewed = store.assign_milk_name(
        "user", original_name="李雷", milk_name="李奶",
        name_options=lambda original: (f"尼古拉斯·{bot.make_milk_name(original)}",),
    )
    assert renewed.original_name == initial.original_name
    assert renewed.milk_name in {
        f"尼古拉斯·{variant}" for variant in bot.milk_nickname_variants("王小明")
    }
    restarted = BotStore(tmp_path)
    assert restarted.milk_name_for_user("user") == renewed
    assert restarted.assign_milk_name(
        "user", original_name="赵小李", milk_name="赵小奶"
    ) == renewed
    assert not restarted.claim_milk_name_welcome("chat", "user")


def test_nickname_generator_can_choose_each_character_position(monkeypatch):
    variants = bot.milk_nickname_variants("韩婉清")
    assert set(variants) == {"奶婉清", "韩奶清", "韩婉奶", "奶韩婉清", "韩婉清奶"}
    for selected in variants:
        monkeypatch.setattr(bot.secrets, "choice", lambda choices, selected=selected: selected)
        assert bot.make_milk_name("韩婉清") == selected


def test_random_welcome_factory_does_not_replace_name_on_retry_or_across_groups(tmp_path):
    store = BotStore(tmp_path)
    assigned = store.assign_milk_name(
        "user", original_name="韩婉清", milk_name="韩婉奶",
        name_options=lambda original: ("尼古拉斯·奶",), replace_existing=False,
    )
    assert assigned.milk_name == "尼古拉斯·奶"
    def should_not_generate(original):
        raise AssertionError("Welcome must keep the saved name")
    assert store.assign_milk_name(
        "user", original_name="韩婉清", milk_name="韩婉奶",
        name_options=should_not_generate, replace_existing=False,
    ) == assigned


def test_simultaneous_requests_from_same_user_serialize_against_latest_name(tmp_path):
    store = BotStore(tmp_path)
    initial = store.assign_milk_name("user", original_name="王小明", milk_name="王小奶")
    transitions = []
    candidates = iter(f"称号{index}·奶" for index in range(12))

    def options(original):
        assert original == initial.original_name
        previous = store.milk_name_for_user("user").milk_name
        name = next(candidates)
        transitions.append((previous, name))
        return (name,)

    def renew(_):
        return store.assign_milk_name(
            "user", original_name="李雷", milk_name="李奶", name_options=options
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(renew, range(12)))
    assert len({result.milk_name for result in results}) == 12
    previous = initial.milk_name
    for old, new in transitions:
        assert old == previous
        assert new != old
        previous = new
    assert store.milk_name_for_user("user").milk_name == previous


def test_same_nickname_across_platforms_cannot_claim_same_current_name(tmp_path):
    store = BotStore(tmp_path)
    def assign(identity):
        platform, index = identity
        user_id = IdentityScope(platform, "test").key("user", str(index))
        return store.assign_milk_name(
            user_id, original_name="王小明", milk_name="王小奶",
            name_options=lambda original: ("尼古拉斯·奶", "最奶的 王小奶"),
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(assign, product(("qq", "feishu"), range(16))))
    assert len({result.milk_name for result in results}) == 32
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM milk_names").fetchone()[0] == 32
        assert connection.execute(
            "SELECT COUNT(DISTINCT milk_name) FROM milk_names"
        ).fetchone()[0] == 32
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE milk_names SET milk_name = '同一个奶名'")


def test_occupied_and_previous_names_are_skipped_before_random_fallback(tmp_path, monkeypatch):
    store = BotStore(tmp_path)
    store.assign_milk_name("other", original_name="李雷", milk_name="尼古拉斯·奶")
    store.assign_milk_name("user", original_name="王小明", milk_name="王小奶")
    renewed = store.assign_milk_name(
        "user", original_name="王小明", milk_name="王小奶",
        name_options=lambda original: ("", " 王小奶 ", "尼古拉斯·奶", "最奶的 王小奶"),
    )
    assert renewed.milk_name == "最奶的 王小奶"
    tokens = iter(("taken", "new"))
    store.assign_milk_name("third", original_name="赵小李", milk_name="王小奶·taken")
    monkeypatch.setattr("nailong_classifier.storage.secrets.token_hex", lambda _: next(tokens))
    renewed = store.assign_milk_name(
        "user", original_name="王小明", milk_name="王小奶",
        name_options=lambda original: ("最奶的 王小奶", "尼古拉斯·奶"),
    )
    assert renewed.milk_name == "王小奶·new"


def test_legacy_duplicate_migration_preserves_original_identity_and_is_idempotent(tmp_path):
    store = BotStore(tmp_path)
    first = store.assign_milk_name("first", original_name="王小明", milk_name="王小奶")
    store.assign_milk_name("second", original_name="王小明", milk_name="王小奶")
    unaffected = store.assign_milk_name("third", original_name="李雷", milk_name="李奶")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP INDEX idx_milk_names_unique")
        connection.execute(
            "UPDATE milk_names SET milk_name = '王小奶' WHERE user_hash = ?",
            (store.identity_hash("second"),),
        )
    restarted = BotStore(tmp_path)
    assert restarted.milk_name_for_user("first") == first
    second = restarted.milk_name_for_user("second")
    assert second.original_name == "王小明"
    assert second.milk_name != first.milk_name
    assert restarted.milk_name_for_user("third") == unaffected
    assert BotStore(tmp_path).milk_name_for_user("second") == second
