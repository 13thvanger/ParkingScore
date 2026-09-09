from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path, PurePosixPath

from .criteria import CriteriaSet, Criterion
from .models import (
    Assessment,
    AssessmentLogUpdate,
    Observation,
    OutputUpdate,
    PhotoMetadata,
    PlateBox,
    RemoteFile,
    RemotePair,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _tsv_value(value: str) -> str:
    return " ".join(value.replace("\t", " ").splitlines()).strip()


def _utc_z(value: str) -> str:
    return from_iso(value).isoformat().replace("+00:00", "Z")


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS remote_files (
                path TEXT PRIMARY KEY,
                size INTEGER,
                modified TEXT,
                signature TEXT NOT NULL,
                stable_polls INTEGER NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                directory TEXT NOT NULL,
                stem TEXT NOT NULL,
                image_path TEXT NOT NULL UNIQUE,
                xml_path TEXT NOT NULL,
                pair_signature TEXT NOT NULL,
                capture_id TEXT NOT NULL,
                plate TEXT NOT NULL,
                place TEXT NOT NULL,
                camera TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                plate_x1 INTEGER,
                plate_y1 INTEGER,
                plate_x2 INTEGER,
                plate_y2 INTEGER,
                group_key TEXT NOT NULL,
                series_id TEXT,
                cache_image_path TEXT NOT NULL,
                eligible INTEGER NOT NULL DEFAULT 0,
                probability INTEGER,
                criteria_details TEXT,
                comment TEXT,
                raw_response TEXT,
                criteria_hash TEXT,
                assessed_at TEXT,
                needs_new_assessment INTEGER NOT NULL DEFAULT 1,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                attempt_criteria_hash TEXT,
                retry_after TEXT,
                failed_criteria_hash TEXT,
                last_error TEXT,
                published_content TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_observations_series
                ON observations(series_id, captured_at);
            CREATE INDEX IF NOT EXISTS idx_observations_work
                ON observations(needs_new_assessment, criteria_hash, retry_after);

            CREATE TABLE IF NOT EXISTS pair_filters (
                image_path TEXT PRIMARY KEY,
                pair_signature TEXT NOT NULL,
                filter_value TEXT,
                eligible INTEGER NOT NULL,
                checked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS progress_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                criteria_hash TEXT NOT NULL,
                ftp_total_pairs INTEGER NOT NULL,
                ftp_stable_pairs INTEGER NOT NULL,
                discovered_total INTEGER NOT NULL,
                assessed_current INTEGER NOT NULL,
                awaiting_assessment INTEGER NOT NULL,
                failed_current INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_progress_snapshots_recorded
                ON progress_snapshots(recorded_at);

            CREATE TABLE IF NOT EXISTS assessment_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id INTEGER NOT NULL,
                directory TEXT NOT NULL,
                stem TEXT NOT NULL,
                assessed_at TEXT NOT NULL,
                probability INTEGER NOT NULL,
                criteria_hash TEXT NOT NULL,
                best INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'live',
                UNIQUE(observation_id, assessed_at, criteria_hash),
                FOREIGN KEY(observation_id) REFERENCES observations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_assessment_events_time
                ON assessment_events(assessed_at, id);

            CREATE TABLE IF NOT EXISTS assessment_log_publications (
                log_date TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                published_at TEXT NOT NULL
            );
            """
        )
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(observations)")
        }
        if "eligible" not in columns:
            self.connection.execute(
                "ALTER TABLE observations "
                "ADD COLUMN eligible INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_observations_eligible_work
            ON observations(eligible, needs_new_assessment, criteria_hash, retry_after)
            """
        )
        filter_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(pair_filters)")
        }
        if "filter_value" not in filter_columns:
            self.connection.execute(
                "ALTER TABLE pair_filters ADD COLUMN filter_value TEXT"
            )
        event_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(assessment_events)")
        }
        if "source" not in event_columns:
            self.connection.execute(
                "ALTER TABLE assessment_events "
                "ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy'"
            )

        observation_columns = {
            "source_key": "TEXT",
            "equipment_serial": "TEXT",
            "sign": "TEXT",
            "canonical_sign": "TEXT",
            "send_probability": "INTEGER",
            "lawn_probability": "INTEGER",
            "evidence_quality_probability": "INTEGER",
            "target_identity_probability": "INTEGER",
            "assessment_id": "TEXT",
            "criteria_version": "TEXT",
            "prompt_version": "TEXT",
            "model_name": "TEXT",
            "current_best": "INTEGER NOT NULL DEFAULT 0",
            "pending_assessment_id": "TEXT",
            "pending_started_at": "TEXT",
            "pending_criteria_hash": "TEXT",
        }
        for name, declaration in observation_columns.items():
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE observations ADD COLUMN {name} {declaration}"
                )

        event_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(assessment_events)")
        }
        assessment_event_columns = {
            "assessment_id": "TEXT",
            "source_key": "TEXT",
            "capture_id": "TEXT",
            "pair_signature": "TEXT",
            "started_at": "TEXT",
            "completed_at": "TEXT",
            "send_probability": "INTEGER",
            "lawn_probability": "INTEGER",
            "evidence_quality_probability": "INTEGER",
            "target_identity_probability": "INTEGER",
            "criteria_version": "TEXT",
            "criteria_snapshot_json": "TEXT",
            "model_name": "TEXT",
            "prompt_version": "TEXT",
            "model_parameters_json": "TEXT",
            "criteria_details_json": "TEXT",
            "comment": "TEXT",
            "raw_response": "TEXT",
            "series_id": "TEXT",
            "mode": "TEXT NOT NULL DEFAULT 'legacy'",
            "image_sha256": "TEXT",
            "evaluation_run_id": "TEXT",
        }
        for name, declaration in assessment_event_columns.items():
            if name not in event_columns:
                self.connection.execute(
                    f"ALTER TABLE assessment_events ADD COLUMN {name} {declaration}"
                )

        # A previous aligned version may already have installed the append-only
        # guards. Temporarily remove them while idempotent legacy backfills run.
        self.connection.executescript(
            """
            DROP TRIGGER IF EXISTS assessment_events_no_update;
            DROP TRIGGER IF EXISTS assessment_events_no_delete;
            """
        )
        if "probability" in columns:
            self.connection.execute(
                "UPDATE observations SET send_probability=probability "
                "WHERE send_probability IS NULL AND probability IS NOT NULL"
            )
        self.connection.execute(
            "UPDATE assessment_events SET send_probability=probability "
            "WHERE send_probability IS NULL"
        )
        self.connection.execute(
            "UPDATE assessment_events SET completed_at=assessed_at "
            "WHERE completed_at IS NULL"
        )
        self.connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_assessment_events_uuid
                ON assessment_events(assessment_id)
                WHERE assessment_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_assessment_events_cursor
                ON assessment_events(id);

            CREATE TABLE IF NOT EXISTS criteria_versions (
                criteria_hash TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                prompt_version TEXT NOT NULL,
                normalized_json TEXT NOT NULL,
                source_text TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                activated_at TEXT,
                retired_at TEXT
            );

            CREATE TABLE IF NOT EXISTS assessment_publications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id TEXT,
                observation_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                remote_path TEXT,
                content_hash TEXT,
                best INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                FOREIGN KEY(observation_id) REFERENCES observations(id)
            );

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                evaluation_run_id TEXT PRIMARY KEY,
                criteria_hash TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS evaluation_items (
                evaluation_run_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                capture_id TEXT NOT NULL,
                assessment_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(evaluation_run_id, source_key),
                FOREIGN KEY(evaluation_run_id)
                    REFERENCES evaluation_runs(evaluation_run_id)
            );

            CREATE TRIGGER IF NOT EXISTS assessment_events_no_update
            BEFORE UPDATE ON assessment_events
            BEGIN
                SELECT RAISE(ABORT, 'assessment_events is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS assessment_events_no_delete
            BEFORE DELETE ON assessment_events
            BEGIN
                SELECT RAISE(ABORT, 'assessment_events is append-only');
            END;
            """
        )
        self.connection.commit()

    def update_remote_files(
        self, files: Iterable[RemoteFile], now: datetime | None = None
    ) -> dict[str, int]:
        now = now or utc_now()
        now_text = to_iso(now)
        existing = {
            row["path"]: row
            for row in self.connection.execute(
                "SELECT path, signature, stable_polls FROM remote_files"
            )
        }
        stable: dict[str, int] = {}
        with self.connection:
            for remote in files:
                previous = existing.get(remote.path)
                count = (
                    int(previous["stable_polls"]) + 1
                    if previous is not None
                    and previous["signature"] == remote.signature
                    else 1
                )
                stable[remote.path] = count
                self.connection.execute(
                    """
                    INSERT INTO remote_files (
                        path, size, modified, signature, stable_polls,
                        first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        size=excluded.size,
                        modified=excluded.modified,
                        signature=excluded.signature,
                        stable_polls=excluded.stable_polls,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (
                        remote.path,
                        remote.size,
                        remote.modified,
                        remote.signature,
                        count,
                        now_text,
                        now_text,
                    ),
                )
        return stable

    def reconcile_remote_pairs(
        self, pairs: Iterable[RemotePair], root: str, recursive: bool
    ) -> int:
        """Deactivate missing pairs only within a successfully listed scope."""
        available = {(pair.image.path, pair.xml.path) for pair in pairs}
        scope = PurePosixPath(root)
        missing = []
        for row in self.connection.execute(
            "SELECT id, image_path, xml_path FROM observations"
        ):
            path = PurePosixPath(row["image_path"])
            in_scope = scope in path.parents if recursive else path.parent == scope
            if in_scope and (row["image_path"], row["xml_path"]) not in available:
                missing.append((row["id"], row["image_path"]))
        count = 0
        with self.connection:
            for observation_id, image_path in missing:
                count += self.connection.execute(
                    "UPDATE observations SET eligible=0, series_id=NULL "
                    "WHERE id=? AND eligible=1", (observation_id,),
                ).rowcount
                # A returning identical pair must pass admission again.
                self.connection.execute(
                    "DELETE FROM pair_filters WHERE image_path=?", (image_path,)
                )
        return count

    def pair_filter_eligibility(self, pair: RemotePair) -> bool | None:
        row = self.connection.execute(
            """
            SELECT pair_signature, eligible
            FROM pair_filters
            WHERE image_path=?
            """,
            (pair.image.path,),
        ).fetchone()
        if row is None or row["pair_signature"] != pair.signature:
            return None
        return bool(row["eligible"])

    def record_pair_filter(
        self,
        pair: RemotePair,
        filter_value: str | None,
        eligible: bool,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO pair_filters (
                    image_path, pair_signature, filter_value, eligible, checked_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    pair_signature=excluded.pair_signature,
                    filter_value=excluded.filter_value,
                    eligible=excluded.eligible,
                    checked_at=excluded.checked_at
                """,
                (
                    pair.image.path,
                    pair.signature,
                    filter_value,
                    int(eligible),
                    to_iso(now),
                ),
            )
            if not eligible:
                self.connection.execute(
                    """
                    UPDATE observations
                    SET eligible=0, series_id=NULL
                    WHERE image_path=?
                    """,
                    (pair.image.path,),
                )

    def deactivate_observation(self, image_path: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE observations
                SET eligible=0, series_id=NULL
                WHERE image_path=?
                """,
                (image_path,),
            )

    def upsert_observation(
        self,
        pair: RemotePair,
        metadata: PhotoMetadata,
        cache_image_path: Path,
        now: datetime | None = None,
    ) -> tuple[int, bool]:
        now = now or utc_now()
        image_path = PurePosixPath(pair.image.path)
        directory = str(image_path.parent)
        if directory == ".":
            directory = ""
        stem = image_path.stem
        source_key = (
            f"{directory.rstrip('/')}/{stem}"
            if directory not in ("", "/")
            else (f"/{stem}" if directory == "/" else stem)
        )
        box = metadata.plate_box
        existing = self.connection.execute(
            """
            SELECT id, pair_signature, eligible
            FROM observations
            WHERE image_path=?
            """,
            (pair.image.path,),
        ).fetchone()

        values = (
            directory,
            stem,
            source_key,
            pair.xml.path,
            pair.signature,
            metadata.capture_id,
            metadata.plate,
            metadata.place,
            metadata.camera,
            metadata.equipment_serial,
            metadata.sign,
            metadata.canonical_sign,
            to_iso(metadata.captured_at),
            to_iso(now),
            metadata.image_width,
            metadata.image_height,
            box.x1 if box else None,
            box.y1 if box else None,
            box.x2 if box else None,
            box.y2 if box else None,
            metadata.group_key,
            str(cache_image_path),
        )

        with self.connection:
            if existing is None:
                cursor = self.connection.execute(
                    """
                    INSERT INTO observations (
                        directory, stem, source_key, image_path, xml_path,
                        pair_signature,
                        capture_id, plate, place, camera, equipment_serial,
                        sign, canonical_sign, captured_at, discovered_at,
                        image_width, image_height,
                        plate_x1, plate_y1, plate_x2, plate_y2,
                        group_key, cache_image_path, eligible,
                        needs_new_assessment
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, 1, 1
                    )
                    """,
                    (
                        directory,
                        stem,
                        source_key,
                        pair.image.path,
                        pair.xml.path,
                        pair.signature,
                        metadata.capture_id,
                        metadata.plate,
                        metadata.place,
                        metadata.camera,
                        metadata.equipment_serial,
                        metadata.sign,
                        metadata.canonical_sign,
                        to_iso(metadata.captured_at),
                        to_iso(now),
                        metadata.image_width,
                        metadata.image_height,
                        box.x1 if box else None,
                        box.y1 if box else None,
                        box.x2 if box else None,
                        box.y2 if box else None,
                        metadata.group_key,
                        str(cache_image_path),
                    ),
                )
                return int(cursor.lastrowid), True

            if existing["pair_signature"] == pair.signature:
                self.connection.execute(
                    """
                    UPDATE observations SET
                        source_key=?, capture_id=?, plate=?, place=?, camera=?,
                        equipment_serial=?, sign=?, canonical_sign=?, group_key=?,
                        eligible=1
                    WHERE id=?
                    """,
                    (
                        source_key,
                        metadata.capture_id,
                        metadata.plate,
                        metadata.place,
                        metadata.camera,
                        metadata.equipment_serial,
                        metadata.sign,
                        metadata.canonical_sign,
                        metadata.group_key,
                        int(existing["id"]),
                    ),
                )
                return int(existing["id"]), False

            self.connection.execute(
                """
                UPDATE observations SET
                    directory=?, stem=?, source_key=?, xml_path=?, pair_signature=?,
                    capture_id=?, plate=?, place=?, camera=?, equipment_serial=?,
                    sign=?, canonical_sign=?, captured_at=?, discovered_at=?,
                    image_width=?, image_height=?, plate_x1=?, plate_y1=?, plate_x2=?,
                    plate_y2=?, group_key=?, cache_image_path=?,
                    eligible=1, needs_new_assessment=1, attempt_count=0,
                    attempt_criteria_hash=NULL, retry_after=NULL,
                    failed_criteria_hash=NULL, last_error=NULL,
                    probability=NULL, send_probability=NULL,
                    lawn_probability=NULL, evidence_quality_probability=NULL,
                    target_identity_probability=NULL, assessment_id=NULL,
                    criteria_hash=NULL, criteria_version=NULL, prompt_version=NULL,
                    model_name=NULL, assessed_at=NULL, criteria_details=NULL,
                    comment=NULL, raw_response=NULL, current_best=0,
                    pending_assessment_id=NULL, pending_started_at=NULL,
                    pending_criteria_hash=NULL
                WHERE id=?
                """,
                (*values, int(existing["id"])),
            )
        return int(existing["id"]), True

    def rebuild_series(self, window_minutes: int) -> None:
        rows = self.connection.execute(
            """
            SELECT id, group_key, captured_at, source_key
            FROM observations
            WHERE eligible=1
            ORDER BY group_key, captured_at, source_key
            """
        ).fetchall()
        assignments: list[tuple[str, int]] = []
        by_group: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            by_group[row["group_key"]].append(row)

        window = timedelta(minutes=window_minutes)
        for group_key, group_rows in by_group.items():
            series_start: datetime | None = None
            series_id = ""
            for row in group_rows:
                captured = from_iso(row["captured_at"])
                if series_start is None or captured - series_start > window:
                    series_start = captured
                    digest_source = f"{group_key}|{to_iso(series_start)}"
                    series_id = hashlib.sha256(digest_source.encode()).hexdigest()[:24]
                assignments.append((series_id, int(row["id"])))

        with self.connection:
            self.connection.execute(
                """
                UPDATE observations
                SET series_id=NULL
                WHERE eligible=0
                """
            )
            self.connection.executemany(
                """
                UPDATE observations
                SET series_id=?
                WHERE id=?
                """,
                assignments,
            )

    def has_pending_new(self, criteria_hash: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM observations
            WHERE eligible=1
              AND (needs_new_assessment=1 OR probability IS NULL)
              AND (failed_criteria_hash IS NULL OR failed_criteria_hash <> ?)
            LIMIT 1
            """,
            (criteria_hash,),
        ).fetchone()
        return row is not None

    def next_new_jobs(
        self, criteria_hash: str, now: datetime, limit: int
    ) -> list[Observation]:
        rows = self.connection.execute(
            """
            SELECT * FROM observations
            WHERE eligible=1
              AND (needs_new_assessment=1 OR probability IS NULL)
              AND (failed_criteria_hash IS NULL OR failed_criteria_hash <> ?)
              AND (retry_after IS NULL OR retry_after <= ?)
            ORDER BY captured_at, discovered_at, image_path
            LIMIT ?
            """,
            (criteria_hash, to_iso(now), limit),
        ).fetchall()
        return [self._observation(row) for row in rows]

    def next_stale_jobs(
        self, criteria_hash: str, now: datetime, limit: int
    ) -> list[Observation]:
        rows = self.connection.execute(
            """
            SELECT * FROM observations
            WHERE eligible=1
              AND needs_new_assessment=0
              AND probability IS NOT NULL
              AND (criteria_hash IS NULL OR criteria_hash <> ?)
              AND (failed_criteria_hash IS NULL OR failed_criteria_hash <> ?)
              AND (retry_after IS NULL OR retry_after <= ?)
            ORDER BY captured_at, discovered_at, image_path
            LIMIT ?
            """,
            (criteria_hash, criteria_hash, to_iso(now), limit),
        ).fetchall()
        return [self._observation(row) for row in rows]

    def begin_assessment(
        self,
        observation_id: int,
        criteria_hash: str,
        now: datetime | None = None,
    ) -> tuple[str, datetime]:
        now = now or utc_now()
        row = self.connection.execute(
            """
            SELECT pending_assessment_id, pending_started_at,
                   pending_criteria_hash
            FROM observations
            WHERE id=?
            """,
            (observation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown observation id: {observation_id}")
        if row["pending_assessment_id"] and row["pending_criteria_hash"] == criteria_hash:
            started = (
                from_iso(str(row["pending_started_at"]))
                if row["pending_started_at"]
                else now
            )
            return str(row["pending_assessment_id"]), started
        assessment_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                """
                UPDATE observations
                SET pending_assessment_id=?, pending_started_at=?,
                    pending_criteria_hash=?
                WHERE id=?
                """,
                (assessment_id, to_iso(now), criteria_hash, observation_id),
            )
        return assessment_id, now

    def register_criteria(
        self,
        criteria: CriteriaSet,
        *,
        activated: bool,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        now_text = to_iso(now)
        with self.connection:
            if activated:
                self.connection.execute(
                    """
                    UPDATE criteria_versions
                    SET retired_at=COALESCE(retired_at, ?)
                    WHERE criteria_hash<>? AND activated_at IS NOT NULL
                      AND retired_at IS NULL
                    """,
                    (now_text, criteria.content_hash),
                )
            self.connection.execute(
                """
                INSERT INTO criteria_versions (
                    criteria_hash, version, schema_version, prompt_version,
                    normalized_json, source_text, first_seen_at, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(criteria_hash) DO UPDATE SET
                    activated_at=CASE
                        WHEN excluded.activated_at IS NOT NULL
                        THEN COALESCE(criteria_versions.activated_at,
                                      excluded.activated_at)
                        ELSE criteria_versions.activated_at
                    END,
                    retired_at=CASE
                        WHEN excluded.activated_at IS NOT NULL THEN NULL
                        ELSE criteria_versions.retired_at
                    END
                """,
                (
                    criteria.content_hash,
                    criteria.version,
                    criteria.schema_version,
                    criteria.prompt_version,
                    criteria.normalized_json,
                    criteria.source_text,
                    now_text,
                    now_text if activated else None,
                ),
            )

    def save_assessment(
        self,
        observation_id: int,
        criteria: str | CriteriaSet,
        assessment: Assessment,
        now: datetime | None = None,
        *,
        assessment_id: str | None = None,
        started_at: datetime | None = None,
        model_name: str = "unknown",
        prompt_version: str = "parking-lawn-v2",
        model_parameters: dict[str, object] | None = None,
        mode: str = "live",
        image_sha256: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> str:
        now = now or utc_now()
        criteria_hash = (
            criteria.content_hash if isinstance(criteria, CriteriaSet) else criteria
        )
        criteria_version = (
            criteria.version if isinstance(criteria, CriteriaSet) else "legacy"
        )
        criteria_snapshot = (
            criteria.normalized_json if isinstance(criteria, CriteriaSet) else None
        )
        if isinstance(criteria, CriteriaSet):
            prompt_version = criteria.prompt_version
        row = self.connection.execute(
            "SELECT * FROM observations WHERE id=?", (observation_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown observation id: {observation_id}")
        assessment_id = assessment_id or (
            str(row["pending_assessment_id"])
            if row["pending_assessment_id"]
            and row["pending_criteria_hash"] == criteria_hash
            else str(uuid.uuid4())
        )
        existing = self.connection.execute(
            "SELECT observation_id FROM assessment_events WHERE assessment_id=?",
            (assessment_id,),
        ).fetchone()
        if existing is not None:
            if int(existing["observation_id"]) != observation_id:
                raise ValueError("assessment_id already belongs to another fact")
            return assessment_id

        started_at = started_at or (
            from_iso(str(row["pending_started_at"]))
            if row["pending_started_at"]
            else now
        )
        send_probability = assessment.send_probability
        lawn_probability = assessment.lawn_probability
        evidence_probability = assessment.evidence_quality_probability
        identity_probability = assessment.target_identity_probability
        if lawn_probability is None:
            lawn_probability = send_probability
        if evidence_probability is None:
            evidence_probability = send_probability
        if identity_probability is None:
            identity_probability = send_probability
        details_json = json.dumps(
            assessment.criteria_details, ensure_ascii=False, sort_keys=True
        )
        model_parameters_json = json.dumps(
            model_parameters or {}, ensure_ascii=False, sort_keys=True
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO assessment_events (
                    assessment_id, observation_id, directory, stem, source_key,
                    capture_id, pair_signature, started_at, completed_at, assessed_at,
                    probability, send_probability, lawn_probability,
                    evidence_quality_probability, target_identity_probability,
                    criteria_hash, criteria_version, criteria_snapshot_json,
                    model_name, prompt_version, model_parameters_json,
                    criteria_details_json, comment, raw_response, series_id,
                    best, source, mode, image_sha256, evaluation_run_id
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, 0, ?, ?, ?, ?
                )
                """,
                (
                    assessment_id,
                    observation_id,
                    row["directory"],
                    row["stem"],
                    row["source_key"],
                    row["capture_id"],
                    row["pair_signature"],
                    to_iso(started_at),
                    to_iso(now),
                    to_iso(now),
                    send_probability,
                    send_probability,
                    lawn_probability,
                    evidence_probability,
                    identity_probability,
                    criteria_hash,
                    criteria_version,
                    criteria_snapshot,
                    model_name,
                    prompt_version,
                    model_parameters_json,
                    details_json,
                    assessment.comment,
                    assessment.raw_response,
                    row["series_id"],
                    mode,
                    mode,
                    image_sha256,
                    evaluation_run_id,
                ),
            )
            if mode == "live":
                self.connection.execute(
                    """
                    UPDATE observations SET
                        probability=?, send_probability=?, lawn_probability=?,
                        evidence_quality_probability=?,
                        target_identity_probability=?, assessment_id=?,
                        criteria_details=?, comment=?, raw_response=?,
                        criteria_hash=?, criteria_version=?, prompt_version=?,
                        model_name=?, assessed_at=?, needs_new_assessment=0,
                        attempt_count=0, attempt_criteria_hash=NULL,
                        retry_after=NULL, failed_criteria_hash=NULL,
                        last_error=NULL, published_content=NULL,
                        pending_assessment_id=NULL, pending_started_at=NULL,
                        pending_criteria_hash=NULL, current_best=0
                    WHERE id=?
                    """,
                    (
                        send_probability,
                        send_probability,
                        lawn_probability,
                        evidence_probability,
                        identity_probability,
                        assessment_id,
                        details_json,
                        assessment.comment,
                        assessment.raw_response,
                        criteria_hash,
                        criteria_version,
                        prompt_version,
                        model_name,
                        to_iso(now),
                        observation_id,
                    ),
                )
        return assessment_id

    def record_failure(
        self,
        observation_id: int,
        criteria_hash: str,
        error: str,
        max_attempts: int,
        retry_base_seconds: int,
        now: datetime | None = None,
        allow_exhaustion: bool = True,
    ) -> bool:
        """Record a processing failure and return True when retries are exhausted."""
        now = now or utc_now()
        row = self.connection.execute(
            """
            SELECT attempt_count, attempt_criteria_hash, needs_new_assessment
            FROM observations WHERE id=?
            """,
            (observation_id,),
        ).fetchone()
        if row is None:
            return True
        attempts = (
            int(row["attempt_count"]) + 1
            if row["attempt_criteria_hash"] == criteria_hash
            else 1
        )
        exhausted = allow_exhaustion and attempts >= max_attempts
        delay = retry_base_seconds * (2 ** min(attempts - 1, 6))
        retry_after = now + timedelta(seconds=delay)
        with self.connection:
            self.connection.execute(
                """
                UPDATE observations SET
                    attempt_count=?, attempt_criteria_hash=?, retry_after=?,
                    failed_criteria_hash=?, needs_new_assessment=?, last_error=?
                WHERE id=?
                """,
                (
                    attempts,
                    criteria_hash,
                    None if exhausted else to_iso(retry_after),
                    criteria_hash if exhausted else None,
                    0 if exhausted else int(row["needs_new_assessment"]),
                    error[:4000],
                    observation_id,
                ),
            )
        return exhausted

    def release_legacy_transient_ai_failures(self, criteria_hash: str) -> int:
        """Requeue AI failures saved before transient errors were distinguished."""
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE observations SET
                    attempt_count=0,
                    attempt_criteria_hash=NULL,
                    retry_after=NULL,
                    failed_criteria_hash=NULL
                WHERE failed_criteria_hash=?
                  AND last_error LIKE 'AI request failed after retries:%'
                """,
                (criteria_hash,),
            )
        return int(cursor.rowcount)

    def output_updates(
        self,
        criteria_hash: str,
        window_minutes: int,
        now: datetime | None = None,
    ) -> list[OutputUpdate]:
        now = now or utc_now()
        rows = self.connection.execute(
            """
            SELECT * FROM observations
            WHERE eligible=1 AND series_id IS NOT NULL
            ORDER BY series_id, captured_at, source_key
            """
        ).fetchall()
        by_series: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            by_series[row["series_id"]].append(row)

        updates: list[OutputUpdate] = []
        quiet_window = timedelta(minutes=window_minutes)
        for series_rows in by_series.values():
            last_discovery = max(from_iso(row["discovered_at"]) for row in series_rows)
            is_open = now - last_discovery < quiet_window
            all_current = all(
                row["probability"] is not None
                and row["criteria_hash"] == criteria_hash
                and not bool(row["needs_new_assessment"])
                and row["assessment_id"] is not None
                and row["lawn_probability"] is not None
                and row["evidence_quality_probability"] is not None
                and row["target_identity_probability"] is not None
                and row["assessed_at"] is not None
                and row["criteria_version"] is not None
                and row["model_name"] is not None
                and row["prompt_version"] is not None
                for row in series_rows
            )

            winner_id: int | None = None
            if not is_open and all_current:
                winner = min(
                    series_rows,
                    key=lambda row: (
                        -int(row["probability"]),
                        row["captured_at"],
                        row["source_key"],
                    ),
                )
                winner_id = int(winner["id"])

            for row in series_rows:
                v2_ready = (
                    row["probability"] is not None
                    and row["assessment_id"] is not None
                    and row["lawn_probability"] is not None
                    and row["evidence_quality_probability"] is not None
                    and row["target_identity_probability"] is not None
                    and row["assessed_at"] is not None
                    and row["criteria_version"] is not None
                    and row["model_name"] is not None
                    and row["prompt_version"] is not None
                )
                if not v2_ready:
                    continue
                publish = False
                best = False
                if is_open:
                    publish = True
                elif all_current:
                    publish = True
                    best = int(row["id"]) == winner_id
                elif (
                    row["published_content"] is None
                    and row["criteria_hash"] == criteria_hash
                    and not bool(row["needs_new_assessment"])
                ):
                    publish = True

                if not publish:
                    continue
                first_lines = (
                    f"probability={int(row['probability'])}\n"
                    f"best={'true' if best else 'false'}\n"
                )
                content = first_lines + (
                    "schema_version=2\n"
                    f"assessment_id={row['assessment_id']}\n"
                    f"capture_id={row['capture_id']}\n"
                    f"lawn_probability={int(row['lawn_probability'])}\n"
                    "evidence_quality_probability="
                    f"{int(row['evidence_quality_probability'])}\n"
                    "target_identity_probability="
                    f"{int(row['target_identity_probability'])}\n"
                    f"assessed_at={_utc_z(str(row['assessed_at']))}\n"
                    f"criteria_hash={row['criteria_hash']}\n"
                    f"criteria_version={row['criteria_version']}\n"
                    f"model={row['model_name']}\n"
                    f"prompt_version={row['prompt_version']}\n"
                )
                if content == row["published_content"]:
                    continue
                directory = row["directory"]
                output_path = (
                    f"{directory.rstrip('/')}/{row['stem']}.txt"
                    if directory not in ("", "/")
                    else (
                        f"/{row['stem']}.txt"
                        if directory == "/"
                        else f"{row['stem']}.txt"
                    )
                )
                updates.append(
                    OutputUpdate(
                        observation_id=int(row["id"]),
                        remote_path=output_path,
                        content=content,
                    )
                )
        return sorted(
            updates,
            key=lambda update: ("best=true" in update.content, update.remote_path),
        )

    def mark_published(
        self,
        observation_id: int,
        content: str,
        remote_path: str | None = None,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        best = "best=true" in content.splitlines()[:2]
        row = self.connection.execute(
            """
            SELECT assessment_id FROM observations WHERE id=?
            """,
            (observation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown observation id: {observation_id}")
        with self.connection:
            self.connection.execute(
                """
                UPDATE observations
                SET published_content=?, current_best=?
                WHERE id=?
                """,
                (content, int(best), observation_id),
            )
            self.connection.execute(
                """
                INSERT INTO assessment_publications (
                    assessment_id, observation_id, created_at, remote_path,
                    content_hash, best, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'published')
                """,
                (
                    row["assessment_id"],
                    observation_id,
                    to_iso(now),
                    remote_path,
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    int(best),
                ),
            )

    def set_latest_assessment_best(
        self, observation_id: int, best: bool
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE observations SET current_best=? WHERE id=?",
                (int(best), observation_id),
            )

    def record_publication_failure(
        self,
        observation_id: int,
        remote_path: str,
        error: str,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        row = self.connection.execute(
            """
            SELECT assessment_id, current_best FROM observations WHERE id=?
            """,
            (observation_id,),
        ).fetchone()
        if row is None:
            return
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO assessment_publications (
                    assessment_id, observation_id, created_at, remote_path,
                    best, status, error
                ) VALUES (?, ?, ?, ?, ?, 'failed', ?)
                """,
                (
                    row["assessment_id"],
                    observation_id,
                    to_iso(now),
                    remote_path,
                    int(row["current_best"]),
                    error[:4000],
                ),
            )

    def assessment_log_updates(
        self, timezone: tzinfo, now: datetime | None = None
    ) -> list[AssessmentLogUpdate]:
        now = now or utc_now()
        rows = self.connection.execute(
            """
            SELECT
                event.id, event.directory, event.stem, event.assessed_at,
                COALESCE(event.send_probability, event.probability) AS probability,
                observation.current_best AS best
            FROM assessment_events AS event
            JOIN observations AS observation
              ON observation.id=event.observation_id
            WHERE observation.eligible=1
              AND event.mode='live'
            ORDER BY event.assessed_at, event.id
            """
        ).fetchall()
        header = (
            "дата оценки\tвремя оценки\tпапка на ftp сервере\t"
            "имя факта\tоценка\tлучший\n"
        )
        lines_by_date: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            assessed = from_iso(str(row["assessed_at"])).astimezone(timezone)
            log_date = assessed.strftime("%d-%m-%Y")
            directory = _tsv_value(str(row["directory"]))
            stem = _tsv_value(str(row["stem"]))
            lines_by_date[log_date].append(
                "\t".join(
                    (
                        log_date,
                        assessed.strftime("%H:%M:%S"),
                        directory,
                        stem,
                        str(int(row["probability"])),
                        "true" if bool(row["best"]) else "false",
                    )
                )
                + "\n"
            )

        published = {
            str(row["log_date"]): str(row["content_hash"])
            for row in self.connection.execute(
                "SELECT log_date, content_hash FROM assessment_log_publications"
            )
        }
        current_log_date = now.astimezone(timezone).strftime("%d-%m-%Y")
        required_dates = set(lines_by_date) | {current_log_date}
        updates: list[AssessmentLogUpdate] = []
        for log_date in sorted(required_dates):
            content = header + "".join(lines_by_date.get(log_date, []))
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if published.get(log_date) == content_hash:
                continue
            updates.append(AssessmentLogUpdate(log_date, content, content_hash))
        return updates

    def mark_assessment_log_published(
        self,
        log_date: str,
        content_hash: str,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO assessment_log_publications (
                    log_date, content_hash, published_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(log_date) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    published_at=excluded.published_at
                """,
                (log_date, content_hash, to_iso(now)),
            )

    def set_meta(self, key: str, value: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

    def ensure_pair_filter_version(self, version: str) -> bool:
        """Invalidate cached decisions when the XML admission rule changes."""
        current = self.get_meta("pair_filter_version")
        if current == version:
            return False
        with self.connection:
            self.connection.execute("DELETE FROM pair_filters")
            self.connection.execute(
                "UPDATE observations SET eligible=0, series_id=NULL"
            )
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES ('pair_filter_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (version,),
            )
        return True

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def export_assessments(
        self, after_id: int = 0, limit: int = 500
    ) -> tuple[list[dict[str, object]], int]:
        if after_id < 0:
            raise ValueError("after_id must be >= 0")
        if not 1 <= limit <= 2000:
            raise ValueError("limit must be between 1 and 2000")
        rows = self.connection.execute(
            """
            SELECT * FROM assessment_events
            WHERE id>?
            ORDER BY id
            LIMIT ?
            """,
            (after_id, limit),
        ).fetchall()
        json_columns = {
            "criteria_snapshot_json": "criteria_snapshot",
            "model_parameters_json": "model_parameters",
            "criteria_details_json": "criteria_details",
        }
        items: list[dict[str, object]] = []
        for row in rows:
            item: dict[str, object] = {
                key: row[key]
                for key in row.keys()  # noqa: SIM118 - sqlite3.Row iterates values
                if key not in json_columns and key not in {"source", "best"}
            }
            item["event_id"] = int(row["id"])
            item.pop("id", None)
            item["best_at_inference"] = bool(row["best"])
            for column, output_name in json_columns.items():
                raw = row[column]
                item[output_name] = json.loads(raw) if raw else None
            items.append(item)
        next_after_id = int(rows[-1]["id"]) if rows else after_id
        return items, next_after_id

    def active_criteria(self) -> dict[str, object] | None:
        row = self.connection.execute(
            """
            SELECT * FROM criteria_versions
            WHERE activated_at IS NOT NULL AND retired_at IS NULL
            ORDER BY activated_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "criteria_hash": row["criteria_hash"],
            "version": row["version"],
            "schema_version": int(row["schema_version"]),
            "prompt_version": row["prompt_version"],
            "criteria": json.loads(row["normalized_json"]),
            "activated_at": row["activated_at"],
        }

    def active_criteria_set(self) -> CriteriaSet | None:
        row = self.connection.execute(
            """
            SELECT * FROM criteria_versions
            WHERE activated_at IS NOT NULL AND retired_at IS NULL
            ORDER BY activated_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        normalized = json.loads(row["normalized_json"])
        definitions = tuple(
            Criterion(
                id=str(item["id"]),
                category=str(item["category"]),
                text=str(item["text"]),
            )
            for item in normalized
        )
        return CriteriaSet(
            items=tuple(item.text for item in definitions),
            content_hash=str(row["criteria_hash"]),
            definitions=definitions,
            schema_version=int(row["schema_version"]),
            version=str(row["version"]),
            prompt_version=str(row["prompt_version"]),
            source_text=str(row["source_text"]),
        )

    def observation_by_source_key(self, source_key: str) -> Observation | None:
        rows = self.connection.execute(
            "SELECT * FROM observations WHERE source_key=? AND eligible=1",
            (source_key,),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError(f"Ambiguous source_key: {source_key}")
        return self._observation(rows[0])

    def begin_evaluation_run(
        self,
        evaluation_run_id: str,
        criteria: CriteriaSet,
        dataset_hash: str,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        existing = self.connection.execute(
            "SELECT * FROM evaluation_runs WHERE evaluation_run_id=?",
            (evaluation_run_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["criteria_hash"] != criteria.content_hash
                or existing["dataset_hash"] != dataset_hash
            ):
                raise ValueError(
                    "evaluation_run_id already exists with different criteria or dataset"
                )
            return
        self.register_criteria(criteria, activated=False, now=now)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO evaluation_runs (
                    evaluation_run_id, criteria_hash, dataset_hash, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    evaluation_run_id,
                    criteria.content_hash,
                    dataset_hash,
                    to_iso(now),
                ),
            )

    def begin_evaluation_item(
        self,
        evaluation_run_id: str,
        source_key: str,
        capture_id: str,
        now: datetime | None = None,
    ) -> tuple[str, str]:
        now = now or utc_now()
        existing = self.connection.execute(
            """
            SELECT assessment_id, capture_id, status
            FROM evaluation_items
            WHERE evaluation_run_id=? AND source_key=?
            """,
            (evaluation_run_id, source_key),
        ).fetchone()
        if existing is not None:
            if existing["capture_id"] != capture_id:
                raise ValueError(
                    "Evaluation item source_key is bound to another capture_id"
                )
            return str(existing["assessment_id"]), str(existing["status"])
        assessment_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO evaluation_items (
                    evaluation_run_id, source_key, capture_id, assessment_id,
                    status, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (
                    evaluation_run_id,
                    source_key,
                    capture_id,
                    assessment_id,
                    to_iso(now),
                ),
            )
        return assessment_id, "pending"

    def finish_evaluation_item(
        self,
        evaluation_run_id: str,
        source_key: str,
        *,
        error: str | None = None,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        status = "failed" if error else "completed"
        with self.connection:
            self.connection.execute(
                """
                UPDATE evaluation_items
                SET status=?, error=?, updated_at=?
                WHERE evaluation_run_id=? AND source_key=?
                """,
                (
                    status,
                    error[:4000] if error else None,
                    to_iso(now),
                    evaluation_run_id,
                    source_key,
                ),
            )

    def finish_evaluation_run(
        self, evaluation_run_id: str, now: datetime | None = None
    ) -> None:
        now = now or utc_now()
        failed = self.connection.execute(
            """
            SELECT 1 FROM evaluation_items
            WHERE evaluation_run_id=? AND status<>'completed' LIMIT 1
            """,
            (evaluation_run_id,),
        ).fetchone()
        if failed is not None:
            return
        with self.connection:
            self.connection.execute(
                """
                UPDATE evaluation_runs SET completed_at=?
                WHERE evaluation_run_id=?
                """,
                (to_iso(now), evaluation_run_id),
            )

    def evaluation_results(
        self, evaluation_run_id: str
    ) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT assessment_id, source_key, capture_id, status, error, updated_at
            FROM evaluation_items
            WHERE evaluation_run_id=?
            ORDER BY source_key
            """,
            (evaluation_run_id,),
        ).fetchall()
        events: list[dict[str, object]] = []
        cursor = 0
        while True:
            batch, next_cursor = self.export_assessments(
                after_id=cursor, limit=2000
            )
            events.extend(batch)
            if len(batch) < 2000:
                break
            cursor = next_cursor
        by_assessment = {
            str(item["assessment_id"]): item
            for item in events
            if item.get("evaluation_run_id") == evaluation_run_id
        }
        return [
            {
                "evaluation_run_id": evaluation_run_id,
                "source_key": row["source_key"],
                "capture_id": row["capture_id"],
                "assessment_id": row["assessment_id"],
                "status": row["status"],
                "error": row["error"],
                "updated_at": row["updated_at"],
                "assessment": by_assessment.get(str(row["assessment_id"])),
            }
            for row in rows
        ]

    def statistics(self, criteria_hash: str) -> dict[str, int]:
        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN probability IS NULL THEN 1 ELSE 0 END) AS unassessed,
                SUM(CASE WHEN probability IS NOT NULL
                              AND (criteria_hash IS NULL OR criteria_hash <> ?)
                         THEN 1 ELSE 0 END) AS stale,
                SUM(CASE WHEN failed_criteria_hash = ? THEN 1 ELSE 0 END) AS failed
            FROM observations
            WHERE eligible=1
            """,
            (criteria_hash, criteria_hash),
        ).fetchone()
        return {
            key: int(row[key] or 0)
            for key in ("total", "unassessed", "stale", "failed")
        }

    def progress_report_due(
        self,
        criteria_hash: str,
        interval_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        now = now or utc_now()
        row = self.connection.execute(
            """
            SELECT recorded_at, criteria_hash
            FROM progress_snapshots
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None or row["criteria_hash"] != criteria_hash:
            return True
        elapsed = now - from_iso(str(row["recorded_at"]))
        return elapsed.total_seconds() >= interval_seconds

    def record_progress_snapshot(
        self,
        criteria_hash: str,
        ftp_total_pairs: int,
        ftp_stable_pairs: int,
        now: datetime | None = None,
    ) -> dict[str, int]:
        now = now or utc_now()
        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS discovered_total,
                SUM(CASE WHEN probability IS NOT NULL
                              AND criteria_hash = ?
                              AND needs_new_assessment = 0
                         THEN 1 ELSE 0 END) AS assessed_current,
                SUM(CASE WHEN failed_criteria_hash = ?
                         THEN 1 ELSE 0 END) AS failed_current
            FROM observations
            WHERE eligible=1
            """,
            (criteria_hash, criteria_hash),
        ).fetchone()
        discovered_total = int(row["discovered_total"] or 0)
        assessed_current = int(row["assessed_current"] or 0)
        failed_current = int(row["failed_current"] or 0)
        awaiting_assessment = max(ftp_total_pairs - assessed_current, 0)
        snapshot = {
            "ftp_total_pairs": ftp_total_pairs,
            "ftp_stable_pairs": ftp_stable_pairs,
            "discovered_total": discovered_total,
            "assessed_current": assessed_current,
            "awaiting_assessment": awaiting_assessment,
            "failed_current": failed_current,
        }
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO progress_snapshots (
                    recorded_at, criteria_hash, ftp_total_pairs,
                    ftp_stable_pairs, discovered_total, assessed_current,
                    awaiting_assessment, failed_current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    to_iso(now),
                    criteria_hash,
                    ftp_total_pairs,
                    ftp_stable_pairs,
                    discovered_total,
                    assessed_current,
                    awaiting_assessment,
                    failed_current,
                ),
            )
        return snapshot

    def _observation(self, row: sqlite3.Row) -> Observation:
        box = None
        if all(
            row[key] is not None
            for key in ("plate_x1", "plate_y1", "plate_x2", "plate_y2")
        ):
            box = PlateBox(
                int(row["plate_x1"]),
                int(row["plate_y1"]),
                int(row["plate_x2"]),
                int(row["plate_y2"]),
            )
        return Observation(
            id=int(row["id"]),
            directory=row["directory"],
            stem=row["stem"],
            source_key=row["source_key"],
            image_path=row["image_path"],
            xml_path=row["xml_path"],
            pair_signature=row["pair_signature"],
            capture_id=row["capture_id"],
            plate=row["plate"],
            place=row["place"],
            camera=row["camera"],
            equipment_serial=row["equipment_serial"],
            sign=row["sign"],
            canonical_sign=row["canonical_sign"],
            captured_at=from_iso(row["captured_at"]),
            discovered_at=from_iso(row["discovered_at"]),
            image_width=row["image_width"],
            image_height=row["image_height"],
            plate_box=box,
            group_key=row["group_key"],
            series_id=row["series_id"],
            send_probability=row["send_probability"],
            lawn_probability=row["lawn_probability"],
            evidence_quality_probability=row["evidence_quality_probability"],
            target_identity_probability=row["target_identity_probability"],
            assessment_id=row["assessment_id"],
            criteria_hash=row["criteria_hash"],
            criteria_version=row["criteria_version"],
            prompt_version=row["prompt_version"],
            model_name=row["model_name"],
            current_best=bool(row["current_best"]),
            needs_new_assessment=bool(row["needs_new_assessment"]),
            pending_assessment_id=row["pending_assessment_id"],
            pending_started_at=(
                from_iso(row["pending_started_at"])
                if row["pending_started_at"]
                else None
            ),
            cache_image_path=Path(row["cache_image_path"]),
        )
