from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import nailong_classifier.bot as bot
from nailong_classifier.adapters.base import (
    IdentityScope,
    ImageTarget,
    Message,
)
from nailong_classifier.adapters.config import AdapterConfig, create_adapter, load_adapter_configs
from nailong_classifier.adapters.feishu import FeishuAdapter
from nailong_classifier.adapters.feishu_process import ProcessFeishuAdapter


def test_identity_namespaces_preserve_legacy_and_reject_cross_bot_routes() -> None:
    scopes = [
        IdentityScope("feishu", "main", legacy=True),
        IdentityScope("feishu", "second"),
        IdentityScope("qq", "qq-a"),
        IdentityScope("qq", "qq-b"),
    ]
    keys = [scope.key("message", "123") for scope in scopes]
    assert len(set(keys)) == 4
    assert keys[0] == "123"
    for scope, key in zip(scopes, keys, strict=True):
        assert scope.raw("message", key) == "123"
        for other in scopes:
            if other is not scope:
                assert not other.owns("message", key)
                with pytest.raises(ValueError, match="another adapter"):
                    other.raw("message", key)


@pytest.mark.parametrize(
    "rows",
    [
        '[[]]',
        '[[bots]]\nid="bad:id"\nplatform="feishu"',
        '[[bots]]\nid="a"\nplatform="unknown"',
        '[[bots]]\nid="a"\nplatform="qq"\nlegacy=true',
        '[[bots]]\nid="a"\nplatform="feishu"\n'
        '[[bots]]\nid="a"\nplatform="feishu"',
        '[[bots]]\nid="a"\nplatform="feishu"\nlegacy=true\n'
        '[[bots]]\nid="b"\nplatform="feishu"\nlegacy=true',
    ],
)
def test_invalid_instance_configs_fail_before_startup(tmp_path, rows) -> None:
    path = tmp_path / "bots.toml"
    path.write_text(rows)
    with pytest.raises(ValueError):
        load_adapter_configs(path)


def test_multiple_feishu_instances_use_process_adapters(monkeypatch) -> None:
    monkeypatch.setenv("TEST_APP", "dummy")
    monkeypatch.setenv("TEST_SECRET", "dummy")
    adapter = create_adapter(
        AdapterConfig("a", "feishu", {"app_id_env": "TEST_APP", "app_secret_env": "TEST_SECRET"}),
        isolate_feishu=True,
    )
    assert isinstance(adapter, ProcessFeishuAdapter)


def test_two_sdk_workers_can_run_and_shutdown_without_shared_global_loop() -> None:
    async def exercise():
        adapters = [
            ProcessFeishuAdapter(IdentityScope("feishu", name), app_id="dummy", app_secret="dummy")
            for name in ("a", "b")
        ]
        try:
            results = await asyncio.gather(*(
                adapter.resolve_image_target(Message(
                    message_id=adapter.scope.key("message", "1"),
                    chat_id=adapter.scope.key("chat", "1"),
                    chat_type="group",
                    sender_id=adapter.scope.key("user", "1"),
                ))
                for adapter in adapters
            ))
            assert results == [ImageTarget(), ImageTarget()]
            pids = {adapter._process.pid for adapter in adapters}
            assert len(pids) == 2
            assert os.getpid() not in pids
        finally:
            await asyncio.gather(*(adapter.close() for adapter in adapters))
        assert all(not adapter._process.is_alive() for adapter in adapters)
    asyncio.run(exercise())


def test_feishu_ids_and_parent_recovery_are_normalized_at_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        "nailong_classifier.adapters.feishu.FeishuChannel",
        lambda **kwargs: SimpleNamespace(client=SimpleNamespace()),
    )
    adapter = FeishuAdapter(IdentityScope("feishu", "a"), app_id="dummy", app_secret="dummy")
    message = adapter.normalize_message(SimpleNamespace(
        message_id="source", chat_id="chat", chat_type="group", sender_id="user",
        sender_name=None, sender_is_bot=False, mentioned_bot=True,
        body_text="识图", content_text="识图", raw_content_type="text",
        reply_to_message_id=None, resources=[], mentions=[],
        raw={"parent_id": "parent"},
    ))
    assert adapter.scope.raw("message", message.reply_to_message_id) == "parent"
    assert message.native is not None
    assert not hasattr(message, "raw")


@pytest.mark.parametrize("platforms", [("feishu",), ("feishu", "qq"), ("feishu", "feishu")])
def test_all_runtimes_receive_the_same_single_loaded_backend(
    tmp_path, monkeypatch, platforms
) -> None:
    configs = [AdapterConfig(f"bot-{index}", platform, {})
               for index, platform in enumerate(platforms)]
    loaded = []
    runtimes = []
    isolation = []

    class Adapter:
        def __init__(self, config):
            self.scope = IdentityScope(config.platform, config.instance_id)
            self.closed = False

        async def close(self):
            self.closed = True

    adapters = [Adapter(config) for config in configs]
    monkeypatch.setattr(bot, "load_adapter_configs", lambda path: configs)
    def create(config, *, isolate_feishu=False):
        isolation.append(isolate_feishu)
        return adapters[configs.index(config)]

    monkeypatch.setattr(bot, "create_adapter", create)
    monkeypatch.setattr(
        bot, "NailongPredictor",
        lambda *args, **kwargs: loaded.append(object()) or SimpleNamespace(),
    )

    async def run(args, adapter, services):
        runtimes.append(services)

    monkeypatch.setattr(bot, "run_adapter", run)
    args = argparse.Namespace(
        bots_config=Path("bots.toml"), model=Path("model.pt"), device="cpu",
        threshold=None, doubao_model=None, state_dir=tmp_path,
    )
    asyncio.run(bot.run_bot(args))
    assert len(loaded) == 1
    assert len(runtimes) == len(configs)
    assert all(runtime is runtimes[0] for runtime in runtimes)
    assert all(isolation)
    assert all(adapter.closed for adapter in adapters)


def test_runtime_failure_still_closes_every_adapter(tmp_path, monkeypatch) -> None:
    configs = [AdapterConfig("a", "feishu", {}), AdapterConfig("b", "qq", {})]
    closed = []

    def adapter(config, **kwargs):
        async def close():
            closed.append(config.instance_id)
        return SimpleNamespace(
            scope=IdentityScope(config.platform, config.instance_id), close=close
        )

    monkeypatch.setattr(bot, "load_adapter_configs", lambda path: configs)
    monkeypatch.setattr(bot, "create_adapter", adapter)
    monkeypatch.setattr(bot, "NailongPredictor", lambda *args, **kwargs: SimpleNamespace())

    async def run(args, adapter, services):
        if adapter.scope.instance_id == "a":
            raise RuntimeError("startup failure")
        await asyncio.Event().wait()

    monkeypatch.setattr(bot, "run_adapter", run)
    args = argparse.Namespace(
        bots_config=Path("bots.toml"), model=Path("model.pt"), device="cpu",
        threshold=None, doubao_model=None, state_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="startup failure"):
        asyncio.run(bot.run_bot(args))
    assert closed == ["a", "b"]
