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
    pending: int
    today_new: int


@dataclass(frozen=True)
class DailyReportTarget:
    chat_id: str
    report_day: str


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
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN label = 'nailong' THEN 1 ELSE 0 END) AS nailong,
                    SUM(CASE WHEN label = 'naiwa' THEN 1 ELSE 0 END) AS naiwa,
                    SUM(CASE WHEN label = 'other' THEN 1 ELSE 0 END) AS pending
                FROM assets
                WHERE origin = 'collected'
                  AND (
                      label IN ('nailong', 'naiwa')
                      OR (
                          corrected = 0
                          AND EXISTS (
                              SELECT 1
                              FROM observations
                              WHERE observations.asset_id = assets.id
                                AND observations.predicted_label
                                    IN ('nailong', 'naiwa')
                          )
                      )
                  )
                """
            ).fetchone()
            today_new = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM assets
                WHERE origin = 'collected'
                  AND first_seen_day = ?
                  AND (
                      label IN ('nailong', 'naiwa')
                      OR (
                          corrected = 0
                          AND EXISTS (
                              SELECT 1
                              FROM observations
                              WHERE observations.asset_id = assets.id
                                AND observations.predicted_label
                                    IN ('nailong', 'naiwa')
                          )
                      )
                  )
                """,
                (day,),
            ).fetchone()["count"]
        return CatalogStats(
            total=counts["total"] or 0,
            nailong=counts["nailong"] or 0,
            naiwa=counts["naiwa"] or 0,
            pending=counts["pending"] or 0,
            today_new=today_new or 0,
        )

    def catalog_assets(self, *, limit: int = 9) -> list[AssetRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assets
                WHERE origin = 'collected'
                  AND (
                      label IN ('nailong', 'naiwa')
                      OR (
                          corrected = 0
                          AND EXISTS (
                              SELECT 1
                              FROM observations
                              WHERE observations.asset_id = assets.id
                                AND observations.predicted_label
                                    IN ('nailong', 'naiwa')
                          )
                      )
                  )
                ORDER BY first_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._asset_from_row(row) for row in rows]

    def _asset_from_row(self, row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            id=row["id"],
            sha256=row["sha256"],
            label=row["label"],
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
