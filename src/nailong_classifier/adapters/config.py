from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BotAdapter, IdentityScope


@dataclass(frozen=True)
class AdapterConfig:
    instance_id: str
    platform: str
    settings: dict[str, Any]
    legacy: bool = False
    catalog_document_id: str | None = None
    catalog_section_block_id: str | None = None


def _environment(settings: dict[str, Any], key: str, *, optional: bool = False) -> str:
    name = settings.get(f"{key}_env")
    value = os.getenv(name, "") if isinstance(name, str) else ""
    if not value and (not optional or name):
        raise ValueError(f"Set the environment variable configured by {key}_env")
    return value


def load_adapter_configs(path: Path | None) -> list[AdapterConfig]:
    """旧部署不配文件仍启动一个飞书实例；多实例凭证只从环境读取。"""
    if path is None:
        return [AdapterConfig(
            "default", "feishu",
            {"app_id_env": "LARK_APP_ID", "app_secret_env": "LARK_APP_SECRET"},
            legacy=True,
        )]
    with path.open("rb") as stream:
        rows = tomllib.load(stream).get("bots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Adapter config must contain at least one [[bots]] entry")
    result = []
    seen = set()
    legacy_seen = False
    documents = set()
    for row in rows:
        instance_id = row.get("id", "")
        platform = row.get("platform")
        legacy = row.get("legacy", False)
        if not isinstance(instance_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", instance_id):
            raise ValueError("Bot id must contain only letters, digits, underscores or hyphens")
        if instance_id in seen:
            raise ValueError("Bot ids must be unique")
        if platform not in {"feishu", "qq"}:
            raise ValueError("Supported platforms: feishu, qq")
        if not isinstance(legacy, bool) or (legacy and (legacy_seen or platform != "feishu")):
            raise ValueError("Only one Feishu bot may own legacy identities")
        document = row.get("catalog_document_id")
        section = row.get("catalog_section_block_id")
        if bool(document) != bool(section):
            raise ValueError("Set both catalog document and section ids")
        if document and (platform != "feishu" or document in documents):
            raise ValueError("Each catalog document must have one Feishu sync owner")
        if document:
            documents.add(document)
        seen.add(instance_id)
        legacy_seen = legacy_seen or legacy
        result.append(AdapterConfig(instance_id, platform, row, legacy, document, section))
    return result


def create_adapter(config: AdapterConfig, *, isolate_feishu: bool = False) -> BotAdapter:
    scope = IdentityScope(config.platform, config.instance_id, legacy=config.legacy)
    if config.platform == "feishu":
        if isolate_feishu:
            from .feishu_process import ProcessFeishuAdapter as FeishuAdapter
        else:
            from .feishu import FeishuAdapter

        return FeishuAdapter(
            scope,
            app_id=_environment(config.settings, "app_id"),
            app_secret=_environment(config.settings, "app_secret"),
        )
    from .qq import QQAdapter

    settings = config.settings
    app_id = settings.get("app_id") or _environment(settings, "app_id")
    if not isinstance(app_id, str) or not app_id.isdigit():
        raise ValueError("QQ app_id must be the official bot AppID")
    for option in ("sandbox", "allow_proactive"):
        if not isinstance(settings.get(option, False), bool):
            raise ValueError(f"QQ {option} must be a boolean")
    return QQAdapter(
        scope,
        app_id=app_id,
        app_secret=_environment(settings, "app_secret"),
        sandbox=settings.get("sandbox", False),
        allow_proactive=settings.get("allow_proactive", False),
    )
