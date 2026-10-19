from __future__ import annotations

import hashlib
import io
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

LABELS = {"nailong", "naiwa", "other"}
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
                    origin TEXT NOT NULL DEFAULT 'collected'
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
                    last_seen_at TEXT NOT NULL
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
        now = datetime.now(LOCAL_TIMEZONE).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assets (
                    sha256, label, confidence, file_name, first_seen_at,
                    first_seen_day, last_seen_at, seen_count, corrected, origin
                )
                VALUES (?, ?, 1.0, ?, ?, NULL, ?, 0, 1, 'seed')
                ON CONFLICT(sha256) DO NOTHING
                """,
                (digest, label, file_name, now, now),
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
    ) -> ObservationRecord:
        self._validate_label(predicted_label)
        now = datetime.now(LOCAL_TIMEZONE)
        now_iso = now.isoformat()
        day = now.date().isoformat()
        digest = hashlib.sha256(image_bytes).hexdigest()
        file_name = self._write_asset(image_bytes, digest)
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
                    first_seen_day, last_seen_at, seen_count, corrected, origin
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, 'collected')
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
                WHERE chat_targets.chat_type IN ('group', 'topic')
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
        )

    @staticmethod
    def _validate_label(label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"Unsupported label: {label}")
