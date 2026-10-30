from __future__ import annotations

import hashlib
import io
import secrets
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from .image_fingerprint import try_content_hash
from .rarity import Rarity, classify_rarity

LABELS = {"nailong", "naiwa", "other"}
LEGACY_NAME_FALLBACKS = {"神秘奶友", "新朋友"}
# 盲盒猜图在奶龙/奶蛙/其他三类之间出题。
GUESS_LABELS = {"nailong", "naiwa", "other"}
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
FORMAT_SUFFIXES = {
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


@dataclass(frozen=True)
class AssetRecord:
    id: int
    sha256: str
    label: str
    confidence: float
    path: Path
    first_seen_at: str
    seen_count: int
    corrected: bool
    origin: str
    # 首次入典时记下的发图者昵称；老库存量图或拿不到名字时为 None，图鉴不显示「首发」行。
    discoverer_name: str | None = None


@dataclass(frozen=True)
class ObservationRecord:
    id: int
    asset: AssetRecord
    is_new_asset: bool
    is_new_observation: bool


@dataclass(frozen=True)
class CorrectionResult:
    observation_id: int
    asset_id: int
    old_label: str
    new_label: str
    asset_path: Path


@dataclass(frozen=True)
class DailyStats:
    total: int
    nailong: int
    naiwa: int
    other: int
    new_nailong: int

    @property
    def milk(self) -> int:
        return self.nailong + self.naiwa

    @property
    def milk_percent(self) -> Decimal:
        if not self.total:
            return Decimal("0.0")
        return (
            Decimal(self.milk) * Decimal("100") / Decimal(self.total)
        ).quantize(Decimal("0.1"))


@dataclass(frozen=True)
class CatalogStats:
    total: int
    nailong: int
    naiwa: int
    today_new: int


@dataclass(frozen=True)
class DailyReportTarget:
    chat_id: str
    report_day: str


@dataclass(frozen=True)
class CatalogDocumentSyncRecord:
    asset: AssetRecord
    status: str
    parent_block_id: str | None
    image_block_id: str | None
    file_token: str | None
    attempt_count: int


@dataclass(frozen=True)
class GuessRound:
    """一局盲盒猜图：出题的图鉴图、正确答案和开局时间。"""

    id: int
    asset: AssetRecord
    answer: str
    started_at: str


@dataclass(frozen=True)
class GuessResult:
    """一次抢答的结果。first_correct 为 True 表示这是本局第一个答对、拿分的人。"""

    round_id: int
    answer: str
    correct: bool
    first_correct: bool
    already_settled: bool
    winner_hash: str | None


@dataclass(frozen=True)
class GuessRanking:
    """猜图积分榜上的一行：玩家名、答对次数、积分。"""

    sender_hash: str
    display_name: str
    wins: int
    score: int


@dataclass(frozen=True)
class HunterRanking:
    """奶龙猎人榜上的一行：发图者昵称、首发入典的奶图数量。"""

    display_name: str
    discovered: int


@dataclass(frozen=True)
class MilkFamilyMember:
    """一个被机器人见过并登记过的奶家人。"""

    display_name: str | None
    first_seen_at: str
    last_seen_at: str


@dataclass(frozen=True)
class MilkName:
    """首次原昵称与最新奶名；日常改昵称和服务重启不改名，明确求赐名才更新。"""

    original_name: str
    milk_name: str
    assigned_at: str
    # 兜底称呼不算真实原昵称；之后得到真实昵称可补全对应关系，不改已赐奶名。
    original_name_known: bool = True


class BotStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.assets_dir = state_dir / "assets"
        self.database_path = state_dir / "bot.sqlite3"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def identity_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    file_name TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    first_seen_day TEXT,
                    last_seen_at TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    corrected INTEGER NOT NULL DEFAULT 0,
                    origin TEXT NOT NULL DEFAULT 'collected',
                    -- 感知指纹：同一张表情不同版本的 phash 相近，用来数“撞脸数”算稀有度。
                    -- 为空表示这张图解析不了，统计时跳过。
                    phash TEXT,
                    -- 首发人：图片首次入典时记下发图者身份哈希和当时昵称（成就署名，定格）。
                    -- 老库存量图这两列为 NULL，图鉴不显示「首发」行，无法补（哈希不可逆）。
                    discoverer_hash TEXT,
                    discoverer_name TEXT
                );

                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY,
                    source_message_id TEXT NOT NULL UNIQUE,
                    response_message_id TEXT UNIQUE,
                    asset_id INTEGER NOT NULL REFERENCES assets(id),
                    chat_hash TEXT NOT NULL,
                    sender_hash TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    final_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accepted INTEGER NOT NULL,
                    observed_at TEXT NOT NULL,
                    observed_day TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_observations_chat_day
                ON observations(chat_hash, observed_day);

                CREATE INDEX IF NOT EXISTS idx_assets_label
                ON assets(label);

                CREATE TABLE IF NOT EXISTS chat_targets (
                    chat_hash TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    chat_type TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    -- 机器人被踢出群后标记为 0，不再向它发送每日播报和主动推送。
                    active INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS daily_reports (
                    chat_hash TEXT NOT NULL REFERENCES chat_targets(chat_hash),
                    report_day TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    response_message_id TEXT,
                    PRIMARY KEY (chat_hash, report_day)
                );

                CREATE TABLE IF NOT EXISTS chain_replies (
                    id INTEGER PRIMARY KEY,
                    chat_hash TEXT NOT NULL,
                    label TEXT NOT NULL,
                    asset_id INTEGER NOT NULL REFERENCES assets(id),
                    response_message_id TEXT NOT NULL UNIQUE,
                    sent_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chain_replies_rotation
                ON chain_replies(chat_hash, label, asset_id, sent_at);

                -- 自动推送抖音链接：发送前先占位，保证并发消息和服务重启后仍按群限制滚动窗口次数。
                CREATE TABLE IF NOT EXISTS video_link_pushes (
                    id INTEGER PRIMARY KEY,
                    chat_hash TEXT NOT NULL,
                    reserved_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_video_link_pushes_window
                ON video_link_pushes(chat_hash, reserved_at);

                CREATE TABLE IF NOT EXISTS catalog_document_sync (
                    document_id TEXT NOT NULL,
                    asset_id INTEGER NOT NULL REFERENCES assets(id),
                    status TEXT NOT NULL,
                    parent_block_id TEXT,
                    image_block_id TEXT,
                    file_token TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    synced_at TEXT,
                    PRIMARY KEY (document_id, asset_id)
                );

                CREATE INDEX IF NOT EXISTS idx_catalog_document_sync_status
                ON catalog_document_sync(document_id, status, updated_at);

                -- 文档图鉴章节里已有图片的内容指纹缓存，按素材 token 记录，避免每轮重复下载。
                -- content_hash 为空表示这张素材解析不了，比较时跳过。
                CREATE TABLE IF NOT EXISTS catalog_document_images (
                    document_id TEXT NOT NULL,
                    file_token TEXT NOT NULL,
                    content_hash TEXT,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, file_token)
                );

                -- 盲盒猜图：每局记录出的图鉴图、正确答案、是否已被抢答，以及第一个答对的人。
                -- winner_hash 非空即代表本局已结算，后来的答对者不再加分。
                CREATE TABLE IF NOT EXISTS guess_rounds (
                    id INTEGER PRIMARY KEY,
                    chat_hash TEXT NOT NULL,
                    asset_id INTEGER NOT NULL REFERENCES assets(id),
                    answer TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    settled_at TEXT,
                    winner_hash TEXT,
                    winner_name TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_guess_rounds_open
                ON guess_rounds(chat_hash, settled_at);

                -- 猜图积分：按群聊 + 玩家累计答对次数，每答对一次 +1 分。
                CREATE TABLE IF NOT EXISTS guess_scores (
                    chat_hash TEXT NOT NULL,
                    sender_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    wins INTEGER NOT NULL DEFAULT 0,
                    score INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (chat_hash, sender_hash)
                );

                -- 奶家人：只保存用户 open_id 的不可逆哈希，不保存原始飞书用户标识。
                -- 用户发过一条可被机器人收到的消息后登记；昵称为空时后续拿到再补。
                CREATE TABLE IF NOT EXISTS milk_family_members (
                    user_hash TEXT PRIMARY KEY,
                    display_name TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                -- 奶名绑定平台适配层的用户身份哈希，不按昵称匹配，避免重名串号。
                -- 原昵称保留赐名时的快照；日常成员昵称更新不会覆盖已赐的奶名。
                CREATE TABLE IF NOT EXISTS milk_names (
                    user_hash TEXT PRIMARY KEY REFERENCES milk_family_members(user_hash),
                    original_name TEXT NOT NULL,
                    milk_name TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    original_name_known INTEGER NOT NULL DEFAULT 1
                );

                -- 入群事件会重投，同一个人在同一个群只起一次奶名。
                CREATE TABLE IF NOT EXISTS milk_name_welcomes (
                    chat_hash TEXT NOT NULL,
                    user_hash TEXT NOT NULL,
                    welcomed_at TEXT NOT NULL,
                    PRIMARY KEY (chat_hash, user_hash)
                );

                -- 默认允许机器人参与群聊；关闭后只响应明确 @ 或命令，不再主动发图、接龙或欢迎新人。
                CREATE TABLE IF NOT EXISTS chat_bot_settings (
                    chat_hash TEXT PRIMARY KEY REFERENCES chat_targets(chat_hash),
                    proactive_enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                """
            )
            sync_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(catalog_document_sync)"
                ).fetchall()
            }
            if "parent_block_id" not in sync_columns:
                connection.execute(
                    """
                    ALTER TABLE catalog_document_sync
                    ADD COLUMN parent_block_id TEXT
                    """
                )
            if "duplicate_of" not in sync_columns:
                connection.execute(
                    """
                    ALTER TABLE catalog_document_sync
                    ADD COLUMN duplicate_of TEXT
                    """
                )
            asset_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(assets)").fetchall()
            }
            if "phash" not in asset_columns:
                connection.execute("ALTER TABLE assets ADD COLUMN phash TEXT")
            if "discoverer_hash" not in asset_columns:
                connection.execute("ALTER TABLE assets ADD COLUMN discoverer_hash TEXT")
            if "discoverer_name" not in asset_columns:
                connection.execute("ALTER TABLE assets ADD COLUMN discoverer_name TEXT")
            chat_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(chat_targets)"
                ).fetchall()
            }
            if "active" not in chat_columns:
                connection.execute(
                    "ALTER TABLE chat_targets ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
            # 旧版本允许同昵称用户撞奶名：只调整重复组里较晚的记录，再建立唯一索引。
            # 迁移和多实例启动共用写锁，不会出现清理完重复、建索引前又插入重复的窗口。
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            name_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(milk_names)")
            }
            if "original_name_known" not in name_columns:
                connection.execute(
                    "ALTER TABLE milk_names ADD COLUMN "
                    "original_name_known INTEGER NOT NULL DEFAULT 1"
                )
                # 老版本只有两个固定兜底称呼，首次迁移时将它们标为待补昵称。
                connection.executemany(
                    "UPDATE milk_names SET original_name_known = 0 WHERE original_name = ?",
                    [(name,) for name in LEGACY_NAME_FALLBACKS],
                )
            connection.execute(
                """
                UPDATE milk_names SET
                    original_name = (
                        SELECT display_name FROM milk_family_members
                        WHERE user_hash = milk_names.user_hash
                    ),
                    original_name_known = 1
                WHERE original_name_known = 0 AND user_hash IN (
                    SELECT user_hash FROM milk_family_members
                    WHERE TRIM(COALESCE(display_name, '')) != ''
                    AND display_name NOT IN ('神秘奶友', '新朋友')
                )
                """
            )
            duplicates = connection.execute(
                """
                SELECT user_hash, milk_name FROM milk_names
                WHERE milk_name IN (
                    SELECT milk_name FROM milk_names GROUP BY milk_name HAVING COUNT(*) > 1
                )
                ORDER BY assigned_at, user_hash
                """
            ).fetchall()
            retained: set[str] = set()
            for row in duplicates:
                name = row["milk_name"]
                if name not in retained:
                    retained.add(name)
                    continue
                unique_name = self._available_milk_name(
                    connection, candidates=(), fallback_name=name
                )
                connection.execute(
                    "UPDATE milk_names SET milk_name = ? WHERE user_hash = ?",
                    (unique_name, row["user_hash"]),
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_milk_names_unique ON milk_names(milk_name)"
            )
        # 老库新增 phash 列后为存量图回填一次指纹，让稀有度统计立刻可用。
        self._backfill_phashes()

    def _backfill_phashes(self) -> None:
        """为还没算过感知指纹的图补算一次。解析不了的图写入空串占位，避免每次启动反复重试。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, file_name FROM assets WHERE phash IS NULL"
            ).fetchall()
            for row in rows:
                path = self.assets_dir / row["file_name"]
                try:
                    phash = try_content_hash(path.read_bytes())
                except OSError:
                    phash = None
                connection.execute(
                    "UPDATE assets SET phash = ? WHERE id = ?",
                    (phash or "", row["id"]),
                )

    @staticmethod
    def _image_suffix(image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format or ""
            image.verify()
        return FORMAT_SUFFIXES.get(image_format.upper(), ".bin")

    def _write_asset(self, image_bytes: bytes, digest: str) -> str:
        suffix = self._image_suffix(image_bytes)
        file_name = f"{digest}{suffix}"
        path = self.assets_dir / file_name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(image_bytes)
            temporary.replace(path)
        return file_name

    def register_seed(self, image_bytes: bytes, label: str) -> AssetRecord:
        self._validate_label(label)
        digest = hashlib.sha256(image_bytes).hexdigest()
        file_name = self._write_asset(image_bytes, digest)
        phash = try_content_hash(image_bytes) or ""
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assets (
                    sha256, label, confidence, file_name, first_seen_at,
                    first_seen_day, last_seen_at, seen_count, corrected, origin,
                    phash
                )
                VALUES (?, ?, 1.0, ?, ?, NULL, ?, 0, 1, 'seed', ?)
                ON CONFLICT(sha256) DO NOTHING
                """,
                (digest, label, file_name, now, now, phash),
            )
            row = connection.execute(
                "SELECT * FROM assets WHERE sha256 = ?",
                (digest,),
            ).fetchone()
        return self._asset_from_row(row)

    def record_observation(
        self,
        *,
        image_bytes: bytes,
        source_message_id: str,
        chat_id: str,
        chat_type: str,
        sender_id: str,
        predicted_label: str,
        confidence: float,
        accepted: bool,
        discoverer_name: str | None = None,
    ) -> ObservationRecord:
        self._validate_label(predicted_label)
        now = datetime.now(LOCAL_TIMEZONE)
        now_iso = now.isoformat()
        day = now.date().isoformat()
        digest = hashlib.sha256(image_bytes).hexdigest()
        file_name = self._write_asset(image_bytes, digest)
        phash = try_content_hash(image_bytes) or ""
        chat_hash = self.identity_hash(chat_id)
        sender_hash = self.identity_hash(sender_id)
        final_label = predicted_label if accepted else "other"

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_targets (
                    chat_hash, chat_id, chat_type, last_seen_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_hash) DO UPDATE SET
                    chat_id = excluded.chat_id,
                    chat_type = excluded.chat_type,
                    last_seen_at = excluded.last_seen_at
                """,
                (chat_hash, chat_id, chat_type, now_iso),
            )
            existing_observation = connection.execute(
                """
                SELECT observations.id AS observation_id, assets.*
                FROM observations
                JOIN assets ON assets.id = observations.asset_id
                WHERE observations.source_message_id = ?
                """,
                (source_message_id,),
            ).fetchone()
            if existing_observation:
                return ObservationRecord(
                    id=existing_observation["observation_id"],
                    asset=self._asset_from_row(existing_observation),
                    is_new_asset=False,
                    is_new_observation=False,
                )

            existing_asset = connection.execute(
                "SELECT id FROM assets WHERE sha256 = ?",
                (digest,),
            ).fetchone()
            is_new_asset = existing_asset is None
            connection.execute(
                """
                INSERT INTO assets (
                    sha256, label, confidence, file_name, first_seen_at,
                    first_seen_day, last_seen_at, seen_count, corrected, origin,
                    phash, discoverer_hash, discoverer_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, 'collected', ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    last_seen_at = excluded.last_seen_at,
                    seen_count = assets.seen_count + 1,
                    label = CASE
                        WHEN assets.corrected = 1 THEN assets.label
                        ELSE excluded.label
                    END,
                    confidence = CASE
                        WHEN assets.corrected = 1 THEN assets.confidence
                        ELSE excluded.confidence
                    END,
                    phash = CASE
                        WHEN assets.phash IS NULL OR assets.phash = ''
                        THEN excluded.phash
                        ELSE assets.phash
                    END
                """,
                (
                    digest,
                    final_label,
                    confidence,
                    file_name,
                    now_iso,
                    day,
                    now_iso,
                    phash,
                    sender_hash,
                    discoverer_name,
                ),
            )
            asset_row = connection.execute(
                "SELECT * FROM assets WHERE sha256 = ?",
                (digest,),
            ).fetchone()
            final_label = asset_row["label"]
            cursor = connection.execute(
                """
                INSERT INTO observations (
                    source_message_id, asset_id, chat_hash, sender_hash,
                    predicted_label, final_label, confidence, accepted,
                    observed_at, observed_day
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_message_id,
                    asset_row["id"],
                    chat_hash,
                    sender_hash,
                    predicted_label,
                    final_label,
                    confidence,
                    int(accepted),
                    now_iso,
                    day,
                ),
            )
            observation_id = int(cursor.lastrowid)

        return ObservationRecord(
            id=observation_id,
            asset=self._asset_from_row(asset_row),
            is_new_asset=is_new_asset,
            is_new_observation=True,
        )

    def refresh_asset_predictions(
        self, predictions: list[tuple[str, str, float, bool]]
    ) -> dict[str, int]:
        """把新 CNN 的判定应用到存量图，不新增发图记录、不改人工确认标签。

        按图片 SHA 匹配，避免下载期间新增素材导致 ID 对错。最新观察保存本次
        模型候选，其余观察保留原候选；最终分类与图鉴一致，沿用人工纠正的
        统计口径。时间、出现次数、发送历史和首发信息均保留。
        """
        for _digest, label, confidence, _accepted in predictions:
            self._validate_label(label)
            if not 0 <= confidence <= 1:
                raise ValueError("Prediction confidence must be between 0 and 1")
        stats: Counter[str] = Counter(
            updated=0, reclassified=0, manual_preserved=0, missing=0
        )
        with self._connect() as connection:
            # 先拿写锁，再检查 corrected；群友同时纠正时不能在检查与更新之间被覆盖。
            connection.execute("BEGIN IMMEDIATE")
            for digest, label, confidence, accepted in predictions:
                asset = connection.execute(
                    "SELECT id, label, corrected FROM assets WHERE sha256 = ?", (digest,)
                ).fetchone()
                if asset is None:
                    stats.update(missing=1)
                    continue
                if asset["corrected"]:
                    stats.update(manual_preserved=1)
                    continue
                final_label = label if accepted else "other"
                connection.execute(
                    "UPDATE assets SET label = ?, confidence = ? WHERE id = ?",
                    (final_label, confidence, asset["id"]),
                )
                connection.execute(
                    """
                    UPDATE observations
                    SET predicted_label = ?, confidence = ?, accepted = ?
                    WHERE id = (SELECT MAX(id) FROM observations WHERE asset_id = ?)
                    """,
                    (label, confidence, int(accepted), asset["id"]),
                )
                connection.execute(
                    "UPDATE observations SET final_label = ? WHERE asset_id = ?",
                    (final_label, asset["id"]),
                )
                stats.update(updated=1)
                if asset["label"] != final_label:
                    stats.update(reclassified=1)
        return dict(stats)

    def attach_response(self, observation_id: int, response_message_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE observations
                SET response_message_id = ?
                WHERE id = ?
                """,
                (response_message_id, observation_id),
            )

    def _apply_label_correction(
        self,
        connection: sqlite3.Connection,
        observation: sqlite3.Row,
        label: str,
    ) -> CorrectionResult:
        connection.execute(
            """
            UPDATE assets
            SET label = ?, corrected = 1, confidence = 1.0
            WHERE id = ?
            """,
            (label, observation["asset_id"]),
        )
        connection.execute(
            """
            UPDATE observations
            SET final_label = ?
            WHERE asset_id = ?
            """,
            (label, observation["asset_id"]),
        )
        return CorrectionResult(
            observation_id=observation["id"],
            asset_id=observation["asset_id"],
            old_label=observation["label"],
            new_label=label,
            asset_path=self.assets_dir / observation["file_name"],
        )

    def correct_label(self, reference_message_id: str, label: str) -> CorrectionResult | None:
        self._validate_label(label)
        with self._connect() as connection:
            observation = connection.execute(
                """
                SELECT observations.id, observations.asset_id, assets.label, assets.file_name
                FROM observations
                JOIN assets ON assets.id = observations.asset_id
                WHERE observations.source_message_id = ?
                   OR observations.response_message_id = ?
                ORDER BY observations.id DESC
                LIMIT 1
                """,
                (reference_message_id, reference_message_id),
            ).fetchone()
            if not observation:
                return None
            return self._apply_label_correction(connection, observation, label)

    def correct_card_label(
        self,
        response_message_id: str,
        observation_id: int,
        label: str,
    ) -> CorrectionResult | None:
        """只纠正与当前卡片消息绑定的识别记录，避免回调值跨卡片误用。"""
        self._validate_label(label)
        with self._connect() as connection:
            observation = connection.execute(
                """
                SELECT observations.id, observations.asset_id, assets.label, assets.file_name
                FROM observations
                JOIN assets ON assets.id = observations.asset_id
                WHERE observations.id = ?
                  AND observations.response_message_id = ?
                LIMIT 1
                """,
                (observation_id, response_message_id),
            ).fetchone()
            if not observation:
                return None
            return self._apply_label_correction(connection, observation, label)

    def observation_by_id(self, observation_id: int) -> ObservationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT assets.*, observations.id AS observation_id
                FROM observations
                JOIN assets ON assets.id = observations.asset_id
                WHERE observations.id = ?
                LIMIT 1
                """,
                (observation_id,),
            ).fetchone()
        if not row:
            return None
        return ObservationRecord(
            id=row["observation_id"],
            asset=self._asset_from_row(row),
            is_new_asset=False,
            is_new_observation=False,
        )

    def random_asset(self, label: str, *, exclude_sha256: str | None = None) -> AssetRecord | None:
        self._validate_label(label)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assets
                WHERE label = ?
                  AND (? IS NULL OR sha256 != ?)
                """,
                (label, exclude_sha256, exclude_sha256),
            ).fetchall()
            if not rows and exclude_sha256:
                rows = connection.execute(
                    "SELECT * FROM assets WHERE label = ?",
                    (label,),
                ).fetchall()
        return self._asset_from_row(secrets.choice(rows)) if rows else None

    def random_milk_asset(self) -> AssetRecord | None:
        """主动推送用：随机取一张奶龙或奶蛙图（含种子图）。

        先在奶龙/奶蛙两类间等概率选一类再抽图，避免某类图多就总发那类；都没有则返回 None。
        按 assets.label 取，收录图和种子图都算。
        """
        with self._connect() as connection:
            by_label: dict[str, list] = {}
            for label in ("nailong", "naiwa"):
                rows = connection.execute(
                    "SELECT * FROM assets WHERE label = ?",
                    (label,),
                ).fetchall()
                if rows:
                    by_label[label] = rows
            if not by_label:
                return None
            label = secrets.choice(sorted(by_label))
        return self._asset_from_row(secrets.choice(by_label[label]))

    def record_milk_family_member(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
    ) -> None:
        """登记一位发过消息的奶家人，保留最早和最近被见到的时间。

        用户标识只以哈希形式落库。昵称是展示用快照，不拿空值覆盖已有昵称。
        """
        if not user_id:
            return
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        name = (display_name.strip() or None) if display_name else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO milk_family_members (
                    user_hash, display_name, first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_hash) DO UPDATE SET
                    display_name = COALESCE(excluded.display_name, display_name),
                    last_seen_at = excluded.last_seen_at
                """,
                (self.identity_hash(user_id), name, now, now),
            )
            if name:
                self._fill_original_name(connection, self.identity_hash(user_id), name)

    def record_milk_family_members(
        self,
        members: list[tuple[str, str | None]],
    ) -> int:
        """批量登记群成员，返回实际写入的非空用户数。"""
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        rows = [
            (
                self.identity_hash(user_id),
                (display_name.strip() or None) if display_name else None,
                now,
                now,
            )
            for user_id, display_name in members
            if user_id
        ]
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO milk_family_members (
                    user_hash, display_name, first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_hash) DO UPDATE SET
                    display_name = COALESCE(excluded.display_name, display_name),
                    last_seen_at = excluded.last_seen_at
                """,
                rows,
            )
            for digest, name, _, _ in rows:
                if name:
                    self._fill_original_name(connection, digest, name)
        return len(rows)

    def remember_milk_family_names(self, members: list[tuple[str, str | None]]) -> None:
        """@ 显示名只补已有成员资料，不把“被提及”当作本人发言或认领奶家人。"""
        with self._connect() as connection:
            for user_id, display_name in members:
                name = display_name.strip() if display_name else ""
                if not user_id or not name:
                    continue
                digest = self.identity_hash(user_id)
                connection.execute(
                    "UPDATE milk_family_members SET display_name = ? WHERE user_hash = ?",
                    (name, digest),
                )
                self._fill_original_name(connection, digest, name)

    @staticmethod
    def _fill_original_name(connection: sqlite3.Connection, digest: str, name: str) -> None:
        """只补系统兜底留下的空缺；正常用户改昵称不覆盖首次原昵称，也不换奶名。"""
        connection.execute(
            """
            UPDATE milk_names SET original_name = ?, original_name_known = 1
            WHERE user_hash = ? AND original_name_known = 0
            """,
            (name, digest),
        )

    @staticmethod
    def _milk_name_from_row(row: sqlite3.Row) -> MilkName:
        return MilkName(
            original_name=row["original_name"], milk_name=row["milk_name"],
            assigned_at=row["assigned_at"], original_name_known=bool(row["original_name_known"]),
        )

    def milk_name_for_user(self, user_id: str) -> MilkName | None:
        """按用户身份查已赐奶名，不把原始平台用户标识写入数据库。"""
        if not user_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM milk_names WHERE user_hash = ?",
                (self.identity_hash(user_id),),
            ).fetchone()
        return self._milk_name_from_row(row) if row is not None else None

    @staticmethod
    def _available_milk_name(
        connection: sqlite3.Connection, *, candidates: Iterable[str],
        fallback_name: str, previous_name: str | None = None,
    ) -> str:
        """在已持有的数据库写锁内查重；词库被占满时追加随机短标记继续查重。"""
        def available(name: str) -> bool:
            return bool(name) and name != previous_name and connection.execute(
                "SELECT 1 FROM milk_names WHERE milk_name = ?", (name,)
            ).fetchone() is None

        for candidate in candidates:
            name = candidate.strip()
            if available(name):
                return name
        # 有限词库或随机反复撞到同一个词时仍能赐名；兜底也须通过查重才能落库。
        for _ in range(64):
            name = f"{fallback_name}·{secrets.token_hex(4)}"
            if available(name):
                return name
        raise RuntimeError("Unable to allocate a unique milk name")

    def assign_milk_name(
        self, user_id: str, *, original_name: str, milk_name: str,
        name_options: Callable[[str], Iterable[str]] | None = None,
        original_name_known: bool = True,
        replace_existing: bool = True,
    ) -> MilkName:
        """原子登记奶名；入群欢迎也随机生成，但 replace_existing=False 保留已有名。

        取得写锁后才读取旧名、生成候选并排除旧名，避免同时求名读到同一旧值。
        候选函数使用库中首次原昵称，昵称改变或并发首授也不会让主体发生漂移。
        所有用户共用同一奶名空间，候选被占用时重抽，数据库唯一索引再兜底。
        返回本次实际保存的映射，之后入群欢迎不会覆盖主动更换过的名字。
        """
        original_name, milk_name = original_name.strip(), milk_name.strip()
        if not user_id or not original_name or not milk_name:
            raise ValueError("User identity, original name and milk name must not be empty")
        digest = self.identity_hash(user_id)
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if original_name_known:
                self._fill_original_name(connection, digest, original_name)
            existing = connection.execute(
                "SELECT original_name, milk_name FROM milk_names WHERE user_hash = ?",
                (digest,),
            ).fetchone()
            should_replace = name_options is not None and replace_existing
            if existing is None or should_replace:
                candidates = (
                    name_options(existing["original_name"] if existing else original_name)
                    if name_options is not None else (milk_name,)
                )
                milk_name = self._available_milk_name(
                    connection,
                    candidates=candidates,
                    fallback_name=milk_name,
                    previous_name=existing["milk_name"] if existing else None,
                )
            connection.execute(
                """
                INSERT INTO milk_family_members (
                    user_hash, display_name, first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_hash) DO UPDATE SET
                    -- 消息处理已更新当前昵称；使用首次原昵称换奶名时不要把它改回去。
                    display_name = COALESCE(display_name, excluded.display_name),
                    last_seen_at = excluded.last_seen_at
                """,
                (digest, original_name, now, now),
            )
            connection.execute(
                """
                INSERT INTO milk_names (
                    user_hash, original_name, milk_name, assigned_at, original_name_known
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_hash) DO NOTHING
                """,
                (digest, original_name, milk_name, now, int(original_name_known)),
            )
            if existing is not None and should_replace:
                connection.execute(
                    "UPDATE milk_names SET milk_name = ?, assigned_at = ? WHERE user_hash = ?",
                    (milk_name, now, digest),
                )
            row = connection.execute(
                "SELECT * FROM milk_names WHERE user_hash = ?",
                (digest,),
            ).fetchone()
        return self._milk_name_from_row(row)

    def claim_milk_name_welcome(self, chat_id: str, user_id: str) -> bool:
        """原子占用一次入群欢迎资格，避免飞书事件重投时重复发消息。"""
        if not chat_id or not user_id:
            return False
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO milk_name_welcomes (chat_hash, user_hash, welcomed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_hash, user_hash) DO NOTHING
                """,
                (
                    self.identity_hash(chat_id),
                    self.identity_hash(user_id),
                    datetime.now(LOCAL_TIMEZONE).isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def release_milk_name_welcome(self, chat_id: str, user_id: str) -> None:
        """欢迎未发送成功时归还本次占用，让后续入群事件仍能重试。"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM milk_name_welcomes WHERE chat_hash = ? AND user_hash = ?",
                (self.identity_hash(chat_id), self.identity_hash(user_id)),
            )

    def milk_family_member(self, user_id: str) -> MilkFamilyMember | None:
        """查询用户是否被机器人登记为奶家人。"""
        if not user_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT display_name, first_seen_at, last_seen_at
                FROM milk_family_members
                WHERE user_hash = ?
                """,
                (self.identity_hash(user_id),),
            ).fetchone()
        if row is None:
            return None
        return MilkFamilyMember(
            display_name=row["display_name"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
        )

    def active_group_chat_ids(self) -> list[str]:
        """返回机器人运行库里已知且仍有效的群聊，用于补登记群成员。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chat_id
                FROM chat_targets
                WHERE chat_type = 'group' AND active = 1
                ORDER BY last_seen_at
                """
            ).fetchall()
        return [str(row["chat_id"]) for row in rows]

    def proactive_enabled(self, chat_id: str) -> bool:
        """未设置的群默认允许主动互动，保持旧群行为不变。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT proactive_enabled
                FROM chat_bot_settings
                WHERE chat_hash = ?
                """,
                (self.identity_hash(chat_id),),
            ).fetchone()
        return row is None or bool(row["proactive_enabled"])

    def set_proactive_enabled(self, chat_id: str, enabled: bool) -> None:
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_bot_settings (chat_hash, proactive_enabled, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_hash) DO UPDATE SET
                    proactive_enabled = excluded.proactive_enabled,
                    updated_at = excluded.updated_at
                """,
                (self.identity_hash(chat_id), int(enabled), now),
            )

    def record_chat(self, chat_id: str, chat_type: str) -> None:
        """记录机器人收到过的会话，让纯文字群也进入后续成员补登记范围。"""
        if not chat_id:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_targets (
                    chat_hash, chat_id, chat_type, last_seen_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_hash) DO UPDATE SET
                    chat_id = excluded.chat_id,
                    chat_type = excluded.chat_type,
                    last_seen_at = excluded.last_seen_at,
                    active = 1
                """,
                (
                    self.identity_hash(chat_id),
                    chat_id,
                    chat_type,
                    datetime.now(LOCAL_TIMEZONE).isoformat(),
                ),
            )

    def next_chain_asset(
        self,
        *,
        chat_id: str,
        label: str,
        exclude_sha256: str | None = None,
    ) -> AssetRecord | None:
        self._validate_label(label)
        chat_hash = self.identity_hash(chat_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT assets.*, MAX(chain_replies.sent_at) AS last_sent_at
                FROM assets
                LEFT JOIN chain_replies
                  ON chain_replies.asset_id = assets.id
                 AND chain_replies.chat_hash = ?
                 AND chain_replies.label = ?
                WHERE assets.label = ?
                  AND (? IS NULL OR assets.sha256 != ?)
                GROUP BY assets.id
                ORDER BY
                    CASE
                        WHEN MAX(chain_replies.sent_at) IS NULL THEN 0
                        ELSE 1
                    END,
                    CASE
                        WHEN MAX(chain_replies.sent_at) IS NULL
                         AND assets.origin = 'collected' THEN 0
                        ELSE 1
                    END,
                    MAX(chain_replies.sent_at) ASC,
                    RANDOM()
                LIMIT 1
                """,
                (
                    chat_hash,
                    label,
                    label,
                    exclude_sha256,
                    exclude_sha256,
                ),
            ).fetchone()
            if row is None and exclude_sha256:
                row = connection.execute(
                    """
                    SELECT assets.*, MAX(chain_replies.sent_at) AS last_sent_at
                    FROM assets
                    LEFT JOIN chain_replies
                      ON chain_replies.asset_id = assets.id
                     AND chain_replies.chat_hash = ?
                     AND chain_replies.label = ?
                    WHERE assets.label = ?
                    GROUP BY assets.id
                    ORDER BY
                        CASE
                            WHEN MAX(chain_replies.sent_at) IS NULL THEN 0
                            ELSE 1
                        END,
                        CASE
                            WHEN MAX(chain_replies.sent_at) IS NULL
                             AND assets.origin = 'collected' THEN 0
                            ELSE 1
                        END,
                        MAX(chain_replies.sent_at) ASC,
                        RANDOM()
                    LIMIT 1
                    """,
                    (chat_hash, label, label),
                ).fetchone()
        return self._asset_from_row(row) if row else None

    def record_chain_reply(
        self,
        *,
        chat_id: str,
        label: str,
        asset_id: int,
        response_message_id: str,
    ) -> None:
        self._validate_label(label)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chain_replies (
                    chat_hash, label, asset_id, response_message_id, sent_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.identity_hash(chat_id),
                    label,
                    asset_id,
                    response_message_id,
                    datetime.now(LOCAL_TIMEZONE).isoformat(),
                ),
            )

    def reserve_video_link_push(
        self,
        chat_id: str,
        *,
        window_seconds: int,
        max_pushes: int,
        now: datetime | None = None,
    ) -> int | None:
        """为一条视频链接占用群聊额度；达到滚动窗口上限时返回 None。

        在发送前占位，避免同一时刻收到多条消息时突破上限。发送失败时调用
        release_video_link_push 释放这次占位。
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_pushes <= 0:
            raise ValueError("max_pushes must be positive")
        reserved_at = now or datetime.now(LOCAL_TIMEZONE)
        cutoff = reserved_at - timedelta(seconds=window_seconds)
        chat_hash = self.identity_hash(chat_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM video_link_pushes WHERE reserved_at <= ?",
                (cutoff.isoformat(),),
            )
            count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM video_link_pushes
                WHERE chat_hash = ?
                  AND reserved_at > ?
                """,
                (chat_hash, cutoff.isoformat()),
            ).fetchone()["count"]
            if count >= max_pushes:
                return None
            cursor = connection.execute(
                """
                INSERT INTO video_link_pushes (chat_hash, reserved_at)
                VALUES (?, ?)
                """,
                (chat_hash, reserved_at.isoformat()),
            )
            return int(cursor.lastrowid)

    def release_video_link_push(self, reservation_id: int) -> None:
        """链接发送失败时释放预留额度。"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM video_link_pushes WHERE id = ?",
                (reservation_id,),
            )

    def daily_stats(self, chat_id: str, *, day: str | None = None) -> DailyStats:
        day = day or datetime.now(LOCAL_TIMEZONE).date().isoformat()
        chat_hash = self.identity_hash(chat_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN final_label = 'nailong' THEN 1 ELSE 0 END) AS nailong,
                    SUM(CASE WHEN final_label = 'naiwa' THEN 1 ELSE 0 END) AS naiwa,
                    SUM(CASE WHEN final_label = 'other' THEN 1 ELSE 0 END) AS other_count
                FROM observations
                WHERE chat_hash = ? AND observed_day = ?
                """,
                (chat_hash, day),
            ).fetchone()
            new_nailong = connection.execute(
                """
                SELECT COUNT(DISTINCT assets.id) AS count
                FROM assets
                JOIN observations ON observations.asset_id = assets.id
                WHERE assets.label = 'nailong'
                  AND assets.origin = 'collected'
                  AND assets.first_seen_day = ?
                  AND observations.chat_hash = ?
                  AND observations.observed_day = ?
                """,
                (day, chat_hash, day),
            ).fetchone()["count"]
        return DailyStats(
            total=row["total"] or 0,
            nailong=row["nailong"] or 0,
            naiwa=row["naiwa"] or 0,
            other=row["other_count"] or 0,
            new_nailong=new_nailong or 0,
        )

    def pending_daily_report_targets(
        self,
        report_day: str,
    ) -> list[DailyReportTarget]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chat_targets.chat_id
                FROM chat_targets
                LEFT JOIN chat_bot_settings
                  ON chat_bot_settings.chat_hash = chat_targets.chat_hash
                WHERE chat_targets.chat_type IN ('group', 'topic')
                  AND chat_targets.active = 1
                  AND COALESCE(chat_bot_settings.proactive_enabled, 1) = 1
                  AND EXISTS (
                      SELECT 1
                      FROM observations
                      WHERE observations.chat_hash = chat_targets.chat_hash
                        AND observations.observed_day = ?
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM daily_reports
                      WHERE daily_reports.chat_hash = chat_targets.chat_hash
                        AND daily_reports.report_day = ?
                  )
                ORDER BY chat_targets.last_seen_at
                """,
                (report_day, report_day),
            ).fetchall()
        return [
            DailyReportTarget(chat_id=row["chat_id"], report_day=report_day)
            for row in rows
        ]

    def deactivate_chat(self, chat_id: str) -> None:
        """机器人被踢出群（发送返回 TARGET_REVOKED）后调用，标记该群失效，不再发送。"""
        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_targets SET active = 0 WHERE chat_hash = ?",
                (self.identity_hash(chat_id),),
            )

    def mark_daily_report_sent(
        self,
        *,
        chat_id: str,
        report_day: str,
        response_message_id: str | None,
    ) -> None:
        chat_hash = self.identity_hash(chat_id)
        sent_at = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO daily_reports (
                    chat_hash, report_day, sent_at, response_message_id
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_hash, report_day) DO NOTHING
                """,
                (chat_hash, report_day, sent_at, response_message_id),
            )

    def catalog_stats(self) -> CatalogStats:
        day = datetime.now(LOCAL_TIMEZONE).date().isoformat()
        with self._connect() as connection:
            counts = connection.execute(
                """
                WITH latest_predictions AS (
                    SELECT
                        observations.asset_id,
                        observations.predicted_label,
                        observations.accepted
                    FROM observations
                    JOIN (
                        SELECT asset_id, MAX(id) AS latest_id
                        FROM observations
                        GROUP BY asset_id
                    ) AS latest
                      ON latest.latest_id = observations.id
                ),
                catalog_candidates AS (
                    SELECT
                        assets.id,
                        assets.first_seen_day,
                        CASE
                            WHEN assets.corrected = 1 THEN assets.label
                            WHEN latest_predictions.accepted = 1
                                THEN latest_predictions.predicted_label
                            ELSE 'other'
                        END AS catalog_label
                    FROM assets
                    LEFT JOIN latest_predictions
                      ON latest_predictions.asset_id = assets.id
                    WHERE assets.origin = 'collected'
                )
                SELECT
                    COUNT(*) AS total,
                    SUM(
                        CASE WHEN catalog_label = 'nailong' THEN 1 ELSE 0 END
                    ) AS nailong,
                    SUM(
                        CASE WHEN catalog_label = 'naiwa' THEN 1 ELSE 0 END
                    ) AS naiwa,
                    SUM(
                        CASE WHEN first_seen_day = ? THEN 1 ELSE 0 END
                    ) AS today_new
                FROM catalog_candidates
                WHERE catalog_label IN ('nailong', 'naiwa')
                """,
                (day,),
            ).fetchone()
        return CatalogStats(
            total=counts["total"] or 0,
            nailong=counts["nailong"] or 0,
            naiwa=counts["naiwa"] or 0,
            today_new=counts["today_new"] or 0,
        )

    def catalog_assets(self, *, limit: int = 9) -> list[AssetRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH latest_predictions AS (
                    SELECT
                        observations.asset_id,
                        observations.predicted_label,
                        observations.accepted
                    FROM observations
                    JOIN (
                        SELECT asset_id, MAX(id) AS latest_id
                        FROM observations
                        GROUP BY asset_id
                    ) AS latest
                      ON latest.latest_id = observations.id
                ),
                catalog_candidates AS (
                    SELECT
                        assets.*,
                        CASE
                            WHEN assets.corrected = 1 THEN assets.label
                            WHEN latest_predictions.accepted = 1
                                THEN latest_predictions.predicted_label
                            ELSE 'other'
                        END AS catalog_label
                    FROM assets
                    LEFT JOIN latest_predictions
                      ON latest_predictions.asset_id = assets.id
                    WHERE assets.origin = 'collected'
                )
                SELECT * FROM catalog_candidates
                WHERE catalog_label IN ('nailong', 'naiwa')
                ORDER BY first_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._asset_from_row(row, label_column="catalog_label")
            for row in rows
        ]

    def asset_rarity(self, asset_id: int) -> Rarity | None:
        """按这张图在群里被发送的累计次数（seen_count）评定稀有度。

        发得越多越烂大街，发得少才稀有。拿不到这张图时返回 None。
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT seen_count FROM assets WHERE id = ?",
                (asset_id,),
            ).fetchone()
        if not row:
            return None
        return classify_rarity(max(1, int(row["seen_count"])))

    def start_guess_round(self, *, chat_id: str) -> GuessRound | None:
        """开一局盲盒猜图：在奶龙/奶蛙/其他三类间等概率选一类，再从该类里随机抽一张当题目。

        先按类别均衡再抽图，避免库存多的「其他」把题目几乎占满；某类没有图就自动跳过。
        没有任何可用图时返回 None。
        """
        chat_hash = self.identity_hash(chat_id)
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH latest_predictions AS (
                    SELECT
                        observations.asset_id,
                        observations.predicted_label,
                        observations.accepted
                    FROM observations
                    JOIN (
                        SELECT asset_id, MAX(id) AS latest_id
                        FROM observations
                        GROUP BY asset_id
                    ) AS latest
                      ON latest.latest_id = observations.id
                ),
                catalog_candidates AS (
                    SELECT
                        assets.*,
                        CASE
                            WHEN assets.corrected = 1 THEN assets.label
                            WHEN latest_predictions.accepted = 1
                                THEN latest_predictions.predicted_label
                            ELSE 'other'
                        END AS catalog_label
                    FROM assets
                    LEFT JOIN latest_predictions
                      ON latest_predictions.asset_id = assets.id
                    WHERE assets.origin = 'collected'
                )
                SELECT * FROM catalog_candidates
                WHERE catalog_label IN ('nailong', 'naiwa', 'other')
                """,
            ).fetchall()
            if not rows:
                return None
            # 先按类别分组，等概率选一类，再在该类里随机抽一张，让三类出现频率接近。
            by_label: dict[str, list] = {}
            for candidate in rows:
                by_label.setdefault(candidate["catalog_label"], []).append(candidate)
            label = secrets.choice(sorted(by_label))
            asset = self._asset_from_row(
                secrets.choice(by_label[label]),
                label_column="catalog_label",
            )
            cursor = connection.execute(
                """
                INSERT INTO guess_rounds (
                    chat_hash, asset_id, answer, started_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (chat_hash, asset.id, asset.label, now),
            )
            round_id = int(cursor.lastrowid)
        return GuessRound(
            id=round_id,
            asset=asset,
            answer=asset.label,
            started_at=now,
        )

    def open_guess_round(self, *, chat_id: str) -> GuessRound | None:
        """返回该群当前未结算的最新一局；没有则返回 None。"""
        chat_hash = self.identity_hash(chat_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT guess_rounds.id AS round_id, guess_rounds.answer,
                       guess_rounds.started_at, assets.*
                FROM guess_rounds
                JOIN assets ON assets.id = guess_rounds.asset_id
                WHERE guess_rounds.chat_hash = ?
                  AND guess_rounds.settled_at IS NULL
                ORDER BY guess_rounds.id DESC
                LIMIT 1
                """,
                (chat_hash,),
            ).fetchone()
            if not row:
                return None
        return GuessRound(
            id=row["round_id"],
            asset=self._asset_from_row(row),
            answer=row["answer"],
            started_at=row["started_at"],
        )

    def close_guess_round(self, *, round_id: int) -> bool:
        """公布答案时关掉一局但不给任何人加分。返回是否确实关掉了一局未结算的局。"""
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE guess_rounds
                SET settled_at = ?
                WHERE id = ? AND settled_at IS NULL
                """,
                (now, round_id),
            )
        return updated.rowcount == 1

    def settle_guess_round(
        self,
        *,
        chat_id: str,
        guess: str,
        sender_id: str,
        display_name: str,
    ) -> GuessResult | None:
        """对该群未结算的最新一局判定一次抢答。

        第一个答对的人拿分并结算本局；之后再答对只回复“已被抢答”。
        没有进行中的局时返回 None。并发时靠 UPDATE ... WHERE settled_at IS NULL 保证只有一人结算。
        """
        self._validate_label(guess)
        chat_hash = self.identity_hash(chat_id)
        sender_hash = self.identity_hash(sender_id)
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            current = connection.execute(
                """
                SELECT id, answer, winner_hash
                FROM guess_rounds
                WHERE chat_hash = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (chat_hash,),
            ).fetchone()
            if not current:
                return None
            if current["winner_hash"] is not None:
                return GuessResult(
                    round_id=current["id"],
                    answer=current["answer"],
                    correct=guess == current["answer"],
                    first_correct=False,
                    already_settled=True,
                    winner_hash=current["winner_hash"],
                )

            answer = current["answer"]
            if guess != answer:
                return GuessResult(
                    round_id=current["id"],
                    answer=answer,
                    correct=False,
                    first_correct=False,
                    already_settled=False,
                    winner_hash=None,
                )

            updated = connection.execute(
                """
                UPDATE guess_rounds
                SET settled_at = ?, winner_hash = ?, winner_name = ?
                WHERE id = ? AND settled_at IS NULL
                """,
                (now, sender_hash, display_name, current["id"]),
            )
            if updated.rowcount != 1:
                # 竞态：别人已经先结算了这一局，这次不再加分。
                winner = connection.execute(
                    "SELECT winner_hash FROM guess_rounds WHERE id = ?",
                    (current["id"],),
                ).fetchone()
                return GuessResult(
                    round_id=current["id"],
                    answer=answer,
                    correct=True,
                    first_correct=False,
                    already_settled=True,
                    winner_hash=winner["winner_hash"] if winner else None,
                )
            connection.execute(
                """
                INSERT INTO guess_scores (
                    chat_hash, sender_hash, display_name, wins, score, updated_at
                )
                VALUES (?, ?, ?, 1, 1, ?)
                ON CONFLICT(chat_hash, sender_hash) DO UPDATE SET
                    display_name = excluded.display_name,
                    wins = guess_scores.wins + 1,
                    score = guess_scores.score + 1,
                    updated_at = excluded.updated_at
                """,
                (chat_hash, sender_hash, display_name, now),
            )
        return GuessResult(
            round_id=current["id"],
            answer=answer,
            correct=True,
            first_correct=True,
            already_settled=False,
            winner_hash=sender_hash,
        )

    def guess_ranking(self, *, chat_id: str, limit: int = 10) -> list[GuessRanking]:
        """该群的猜图积分榜，按积分降序。"""
        chat_hash = self.identity_hash(chat_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sender_hash, display_name, wins, score
                FROM guess_scores
                WHERE chat_hash = ?
                ORDER BY score DESC, wins DESC, updated_at ASC
                LIMIT ?
                """,
                (chat_hash, limit),
            ).fetchall()
        return [
            GuessRanking(
                sender_hash=row["sender_hash"],
                display_name=row["display_name"],
                wins=row["wins"],
                score=row["score"],
            )
            for row in rows
        ]

    def hunter_rankings(self, *, limit: int = 10) -> list[HunterRanking]:
        """奶龙猎人榜：按「首发入典的收藏奶图数」排名。

        只统计有首发人署名、且仍算作收藏奶图（奶龙/奶蛙）的图；老库无署名的图不计入。
        同一人用 discoverer_hash 聚合，名字取最近一次记录到的昵称。
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH latest_predictions AS (
                    SELECT
                        observations.asset_id,
                        observations.predicted_label,
                        observations.accepted
                    FROM observations
                    JOIN (
                        SELECT asset_id, MAX(id) AS latest_id
                        FROM observations
                        GROUP BY asset_id
                    ) AS latest
                      ON latest.latest_id = observations.id
                ),
                catalog_candidates AS (
                    SELECT
                        assets.discoverer_hash,
                        assets.discoverer_name,
                        assets.last_seen_at,
                        CASE
                            WHEN assets.corrected = 1 THEN assets.label
                            WHEN latest_predictions.accepted = 1
                                THEN latest_predictions.predicted_label
                            ELSE 'other'
                        END AS catalog_label
                    FROM assets
                    LEFT JOIN latest_predictions
                      ON latest_predictions.asset_id = assets.id
                    WHERE assets.origin = 'collected'
                      AND assets.discoverer_hash IS NOT NULL
                      AND assets.discoverer_name IS NOT NULL
                )
                SELECT
                    discoverer_hash,
                    COUNT(*) AS discovered,
                    (
                        SELECT inner_c.discoverer_name
                        FROM catalog_candidates AS inner_c
                        WHERE inner_c.discoverer_hash = catalog_candidates.discoverer_hash
                        ORDER BY inner_c.last_seen_at DESC
                        LIMIT 1
                    ) AS display_name
                FROM catalog_candidates
                WHERE catalog_label IN ('nailong', 'naiwa')
                GROUP BY discoverer_hash
                ORDER BY discovered DESC, display_name ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            HunterRanking(
                display_name=row["display_name"],
                discovered=row["discovered"],
            )
            for row in rows
        ]

    def mark_catalog_document_baseline(
        self,
        *,
        document_id: str,
        asset_ids: list[int],
    ) -> int:
        if not asset_ids:
            return 0
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        inserted = 0
        with self._connect() as connection:
            for asset_id in dict.fromkeys(asset_ids):
                cursor = connection.execute(
                    """
                    INSERT INTO catalog_document_sync (
                        document_id, asset_id, status, updated_at, synced_at
                    )
                    SELECT ?, assets.id, 'baseline', ?, ?
                    FROM assets
                    WHERE assets.id = ?
                    ON CONFLICT(document_id, asset_id) DO NOTHING
                    """,
                    (document_id, now, now, asset_id),
                )
                inserted += cursor.rowcount
        return inserted

    def pending_catalog_document_sync(
        self,
        *,
        document_id: str,
        limit: int = 25,
    ) -> list[CatalogDocumentSyncRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH latest_predictions AS (
                    SELECT
                        observations.asset_id,
                        observations.predicted_label,
                        observations.accepted
                    FROM observations
                    JOIN (
                        SELECT asset_id, MAX(id) AS latest_id
                        FROM observations
                        GROUP BY asset_id
                    ) AS latest
                      ON latest.latest_id = observations.id
                ),
                catalog_candidates AS (
                    SELECT
                        assets.*,
                        CASE
                            WHEN assets.corrected = 1 THEN assets.label
                            WHEN latest_predictions.accepted = 1
                                THEN latest_predictions.predicted_label
                            ELSE 'other'
                        END AS catalog_label
                    FROM assets
                    LEFT JOIN latest_predictions
                      ON latest_predictions.asset_id = assets.id
                    WHERE assets.origin = 'collected'
                )
                SELECT
                    catalog_candidates.*,
                    catalog_document_sync.status AS sync_status,
                    catalog_document_sync.parent_block_id,
                    catalog_document_sync.image_block_id,
                    catalog_document_sync.file_token,
                    catalog_document_sync.attempt_count
                FROM catalog_candidates
                LEFT JOIN catalog_document_sync
                  ON catalog_document_sync.asset_id = catalog_candidates.id
                 AND catalog_document_sync.document_id = ?
                WHERE catalog_candidates.catalog_label IN ('nailong', 'naiwa')
                  AND (
                      catalog_document_sync.status IS NULL
                      OR catalog_document_sync.status NOT IN (
                          'baseline', 'synced', 'duplicate'
                      )
                  )
                ORDER BY catalog_candidates.first_seen_at, catalog_candidates.id
                LIMIT ?
                """,
                (document_id, limit),
            ).fetchall()
        return [
            CatalogDocumentSyncRecord(
                asset=self._asset_from_row(row, label_column="catalog_label"),
                status=row["sync_status"] or "pending",
                parent_block_id=row["parent_block_id"],
                image_block_id=row["image_block_id"],
                file_token=row["file_token"],
                attempt_count=row["attempt_count"] or 0,
            )
            for row in rows
        ]

    def mark_catalog_document_target_assigned(
        self,
        *,
        document_id: str,
        asset_id: int,
        parent_block_id: str,
    ) -> None:
        self._upsert_catalog_document_sync(
            document_id=document_id,
            asset_id=asset_id,
            status="target_assigned",
            parent_block_id=parent_block_id,
        )

    def mark_catalog_document_block_created(
        self,
        *,
        document_id: str,
        asset_id: int,
        image_block_id: str,
    ) -> None:
        self._upsert_catalog_document_sync(
            document_id=document_id,
            asset_id=asset_id,
            status="block_created",
            image_block_id=image_block_id,
        )

    def mark_catalog_document_media_uploaded(
        self,
        *,
        document_id: str,
        asset_id: int,
        image_block_id: str,
        file_token: str,
    ) -> None:
        self._upsert_catalog_document_sync(
            document_id=document_id,
            asset_id=asset_id,
            status="media_uploaded",
            image_block_id=image_block_id,
            file_token=file_token,
        )

    def mark_catalog_document_synced(
        self,
        *,
        document_id: str,
        asset_id: int,
        image_block_id: str,
        file_token: str,
    ) -> None:
        self._upsert_catalog_document_sync(
            document_id=document_id,
            asset_id=asset_id,
            status="synced",
            image_block_id=image_block_id,
            file_token=file_token,
            synced=True,
        )

    def mark_catalog_document_duplicate(
        self,
        *,
        document_id: str,
        asset_id: int,
        duplicate_of: str,
    ) -> None:
        # 文档里已有内容相同的图（只是尺寸或编码不同），这张不再追加；
        # duplicate_of 记下文档里对应的素材 token，方便排查。
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalog_document_sync (
                    document_id, asset_id, status, duplicate_of, updated_at
                )
                VALUES (?, ?, 'duplicate', ?, ?)
                ON CONFLICT(document_id, asset_id) DO UPDATE SET
                    status = 'duplicate',
                    duplicate_of = excluded.duplicate_of,
                    last_error = NULL,
                    updated_at = excluded.updated_at
                """,
                (document_id, asset_id, duplicate_of, now),
            )

    def catalog_document_image_hashes(
        self,
        *,
        document_id: str,
    ) -> dict[str, str | None]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT file_token, content_hash
                FROM catalog_document_images
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchall()
        return {row["file_token"]: row["content_hash"] for row in rows}

    def record_catalog_document_image_hash(
        self,
        *,
        document_id: str,
        file_token: str,
        content_hash: str | None,
    ) -> None:
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalog_document_images (
                    document_id, file_token, content_hash, recorded_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id, file_token) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    recorded_at = excluded.recorded_at
                """,
                (document_id, file_token, content_hash, now),
            )

    def mark_catalog_document_sync_failed(
        self,
        *,
        document_id: str,
        asset_id: int,
        error: str,
    ) -> None:
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalog_document_sync (
                    document_id, asset_id, status, attempt_count,
                    last_error, updated_at
                )
                VALUES (?, ?, 'failed', 1, ?, ?)
                ON CONFLICT(document_id, asset_id) DO UPDATE SET
                    status = 'failed',
                    attempt_count = catalog_document_sync.attempt_count + 1,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (document_id, asset_id, error[:2000], now),
            )

    def catalog_document_sync_counts(self, *, document_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM catalog_document_sync
                WHERE document_id = ?
                GROUP BY status
                """,
                (document_id,),
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def _upsert_catalog_document_sync(
        self,
        *,
        document_id: str,
        asset_id: int,
        status: str,
        parent_block_id: str | None = None,
        image_block_id: str | None = None,
        file_token: str | None = None,
        synced: bool = False,
    ) -> None:
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        synced_at = now if synced else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO catalog_document_sync (
                    document_id, asset_id, status, parent_block_id, image_block_id,
                    file_token, updated_at, synced_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, asset_id) DO UPDATE SET
                    status = excluded.status,
                    parent_block_id = COALESCE(
                        excluded.parent_block_id,
                        catalog_document_sync.parent_block_id
                    ),
                    image_block_id = COALESCE(
                        excluded.image_block_id,
                        catalog_document_sync.image_block_id
                    ),
                    file_token = COALESCE(
                        excluded.file_token,
                        catalog_document_sync.file_token
                    ),
                    last_error = NULL,
                    updated_at = excluded.updated_at,
                    synced_at = COALESCE(
                        excluded.synced_at,
                        catalog_document_sync.synced_at
                    )
                """,
                (
                    document_id,
                    asset_id,
                    status,
                    parent_block_id,
                    image_block_id,
                    file_token,
                    now,
                    synced_at,
                ),
            )

    def _asset_from_row(
        self,
        row: sqlite3.Row,
        *,
        label_column: str = "label",
    ) -> AssetRecord:
        return AssetRecord(
            id=row["id"],
            sha256=row["sha256"],
            label=row[label_column],
            confidence=float(row["confidence"]),
            path=self.assets_dir / row["file_name"],
            first_seen_at=row["first_seen_at"],
            seen_count=row["seen_count"],
            corrected=bool(row["corrected"]),
            origin=row["origin"],
            discoverer_name=self._row_value(row, "discoverer_name"),
        )

    @staticmethod
    def _row_value(row: sqlite3.Row, key: str) -> str | None:
        """安全读取某列，列不在本次查询结果里时返回 None（部分查询没 SELECT 该列）。"""
        return row[key] if key in set(row.keys()) else None

    @staticmethod
    def _validate_label(label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"Unsupported label: {label}")
