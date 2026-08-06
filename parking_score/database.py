from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import (
    Assessment,
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

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
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

    def pair_needs_ingest(self, pair: RemotePair) -> bool:
        row = self.connection.execute(
            "SELECT pair_signature FROM observations WHERE image_path = ?",
            (pair.image.path,),
        ).fetchone()
        return row is None or row["pair_signature"] != pair.signature

    def upsert_observation(
        self,
        pair: RemotePair,
        metadata: PhotoMetadata,
        cache_image_path: Path,
        now: datetime | None = None,
    ) -> tuple[int, bool]:
        now = now or utc_now()
        directory = str(Path(pair.image.path).parent).replace("\\", "/")
        if directory == ".":
            directory = ""
        stem = Path(pair.image.path).stem
        box = metadata.plate_box
        existing = self.connection.execute(
            "SELECT id, pair_signature FROM observations WHERE image_path = ?",
            (pair.image.path,),
        ).fetchone()

        values = (
            directory,
            stem,
            pair.xml.path,
            pair.signature,
            metadata.capture_id,
            metadata.plate,
            metadata.place,
            metadata.camera,
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
                        directory, stem, image_path, xml_path, pair_signature,
                        capture_id, plate, place, camera, captured_at,
                        discovered_at, image_width, image_height,
                        plate_x1, plate_y1, plate_x2, plate_y2,
                        group_key, cache_image_path, needs_new_assessment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        directory,
                        stem,
                        pair.image.path,
                        pair.xml.path,
                        pair.signature,
                        metadata.capture_id,
                        metadata.plate,
                        metadata.place,
                        metadata.camera,
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
                return int(existing["id"]), False

            self.connection.execute(
                """
                UPDATE observations SET
                    directory=?, stem=?, xml_path=?, pair_signature=?, capture_id=?,
                    plate=?, place=?, camera=?, captured_at=?, discovered_at=?,
                    image_width=?, image_height=?, plate_x1=?, plate_y1=?,
                    plate_x2=?, plate_y2=?, group_key=?, cache_image_path=?,
                    needs_new_assessment=1, attempt_count=0,
                    attempt_criteria_hash=NULL, retry_after=NULL,
                    failed_criteria_hash=NULL, last_error=NULL
                WHERE id=?
                """,
                (*values, int(existing["id"])),
            )
        return int(existing["id"]), True

    def rebuild_series(self, window_minutes: int) -> None:
        rows = self.connection.execute(
            """
            SELECT id, group_key, captured_at
            FROM observations
            ORDER BY group_key, captured_at, image_path
            """
        ).fetchall()
        assignments: list[tuple[str, int]] = []
        by_group: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            by_group[row["group_key"]].append(row)

        window = timedelta(minutes=window_minutes)
        for group_key, group_rows in by_group.items():
            previous: datetime | None = None
            series_start: datetime | None = None
            series_id = ""
            for row in group_rows:
                captured = from_iso(row["captured_at"])
                if previous is None or captured - previous > window:
                    series_start = captured
                    digest_source = f"{group_key}|{to_iso(series_start)}"
                    series_id = hashlib.sha256(digest_source.encode()).hexdigest()[:24]
                assignments.append((series_id, int(row["id"])))
                previous = captured

        with self.connection:
            self.connection.executemany(
                "UPDATE observations SET series_id=? WHERE id=?", assignments
            )

    def has_pending_new(self, criteria_hash: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM observations
            WHERE (needs_new_assessment=1 OR probability IS NULL)
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
            WHERE (needs_new_assessment=1 OR probability IS NULL)
              AND (failed_criteria_hash IS NULL OR failed_criteria_hash <> ?)
              AND (retry_after IS NULL OR retry_after <= ?)
            ORDER BY discovered_at, captured_at, image_path
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
            WHERE needs_new_assessment=0
              AND probability IS NOT NULL
              AND (criteria_hash IS NULL OR criteria_hash <> ?)
              AND (failed_criteria_hash IS NULL OR failed_criteria_hash <> ?)
              AND (retry_after IS NULL OR retry_after <= ?)
            ORDER BY series_id, captured_at, image_path
            LIMIT ?
            """,
            (criteria_hash, criteria_hash, to_iso(now), limit),
        ).fetchall()
        return [self._observation(row) for row in rows]

    def save_assessment(
        self,
        observation_id: int,
        criteria_hash: str,
        assessment: Assessment,
        now: datetime | None = None,
    ) -> None:
        now = now or utc_now()
        with self.connection:
            self.connection.execute(
                """
                UPDATE observations SET
                    probability=?, criteria_details=?, comment=?, raw_response=?,
                    criteria_hash=?, assessed_at=?, needs_new_assessment=0,
                    attempt_count=0, attempt_criteria_hash=NULL, retry_after=NULL,
                    failed_criteria_hash=NULL, last_error=NULL
                WHERE id=?
                """,
                (
                    assessment.probability,
                    json.dumps(assessment.criteria_details, ensure_ascii=False),
                    assessment.comment,
                    assessment.raw_response,
                    criteria_hash,
                    to_iso(now),
                    observation_id,
                ),
            )

    def record_failure(
        self,
        observation_id: int,
        criteria_hash: str,
        error: str,
        max_attempts: int,
        retry_base_seconds: int,
        now: datetime | None = None,
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
        exhausted = attempts >= max_attempts
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
            WHERE series_id IS NOT NULL
            ORDER BY series_id, captured_at, image_path
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
                for row in series_rows
            )

            winner_id: int | None = None
            if not is_open and all_current:
                winner = min(
                    series_rows,
                    key=lambda row: (
                        -int(row["probability"]),
                        row["captured_at"],
                        row["image_path"],
                    ),
                )
                winner_id = int(winner["id"])

            for row in series_rows:
                if row["probability"] is None:
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
                content = (
                    f"probability={int(row['probability'])}\n"
                    f"best={'true' if best else 'false'}\n"
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

    def mark_published(self, observation_id: int, content: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE observations SET published_content=? WHERE id=?",
                (content, observation_id),
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

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

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
            """,
            (criteria_hash, criteria_hash),
        ).fetchone()
        return {
            key: int(row[key] or 0)
            for key in ("total", "unassessed", "stale", "failed")
        }

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
            image_path=row["image_path"],
            xml_path=row["xml_path"],
            pair_signature=row["pair_signature"],
            plate=row["plate"],
            place=row["place"],
            camera=row["camera"],
            captured_at=from_iso(row["captured_at"]),
            discovered_at=from_iso(row["discovered_at"]),
            image_width=row["image_width"],
            image_height=row["image_height"],
            plate_box=box,
            group_key=row["group_key"],
            series_id=row["series_id"],
            probability=row["probability"],
            criteria_hash=row["criteria_hash"],
            needs_new_assessment=bool(row["needs_new_assessment"]),
            cache_image_path=Path(row["cache_image_path"]),
        )
