import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from parking_score.criteria import CriteriaSet
from parking_score.database import Repository
from parking_score.models import Assessment, PhotoMetadata, RemoteFile, RemotePair


def _add(
    repository: Repository,
    name: str,
    captured_at: datetime,
    discovered_at: datetime,
    *,
    place: str = "test address",
    equipment_serial: str = "camera-1",
) -> int:
    pair = RemotePair(
        RemoteFile(f"/camera/{name}.jpg", 100, "20260801000000"),
        RemoteFile(f"/camera/{name}.xml", 50, "20260801000000"),
    )
    metadata = PhotoMetadata(
        capture_id=name,
        plate="O716MP48",
        place=place,
        camera="camera-1",
        equipment_serial=equipment_serial,
        captured_at=captured_at,
        image_width=1920,
        image_height=1200,
        plate_box=None,
        group_key=f"O716MP48\x1f{equipment_serial}",
    )
    observation_id, _ = repository.upsert_observation(
        pair, metadata, Path(f"/tmp/{name}.jpg"), now=discovered_at
    )
    return observation_id


def _assessment(probability: int) -> Assessment:
    return Assessment(probability, [], "", "{}")


def test_production_v1_database_migrates_to_contract_free_v2(tmp_path) -> None:
    database = tmp_path / "v1.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE observations (
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
        CREATE TABLE assessment_events (
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
        INSERT INTO observations (
            directory, stem, image_path, xml_path, pair_signature, capture_id,
            plate, place, camera, captured_at, discovered_at, group_key,
            cache_image_path, eligible, probability, criteria_hash, assessed_at,
            needs_new_assessment
        ) VALUES (
            '/camera', 'legacy', '/camera/legacy.jpg', '/camera/legacy.xml',
            'image=100:old;xml=50:old', 'legacy-capture', 'A001AA48',
            'legacy place', 'legacy-camera', '2026-08-01T10:00:00+00:00',
            '2026-08-01T10:01:00+00:00', 'A001AA48\u001flegacy-camera',
            '/tmp/legacy.jpg', 1, 55, 'legacy-criteria',
            '2026-08-01T10:02:00+00:00', 0
        );
        INSERT INTO assessment_events (
            observation_id, directory, stem, assessed_at, probability,
            criteria_hash, best, source
        ) VALUES (
            1, '/camera', 'legacy', '2026-08-01T10:02:00+00:00', 55,
            'legacy-criteria', 1, 'live'
        );
        """
    )
    connection.close()

    repository = Repository(database)
    try:
        observation = repository.connection.execute(
            "SELECT probability, send_probability FROM observations WHERE id=1"
        ).fetchone()
        event = repository.connection.execute(
            "SELECT probability, send_probability FROM assessment_events WHERE id=1"
        ).fetchone()
        assert tuple(observation) == (55, 55)
        assert tuple(event) == (55, 55)

        for table in (
            "observations",
            "assessment_events",
            "assessment_publications",
        ):
            columns = {
                row["name"]
                for row in repository.connection.execute(f"PRAGMA table_info({table})")
            }
            assert "series_contract_version" not in columns

        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            repository.connection.execute(
                "UPDATE assessment_events SET probability=99 WHERE id=1"
            )
    finally:
        repository.close()


def test_pair_filter_version_invalidates_cached_decisions(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 19, tzinfo=UTC)
    pair = RemotePair(
        RemoteFile("/camera/a.jpg", 100, "20260819000000"),
        RemoteFile("/camera/a.xml", 50, "20260819000000"),
    )
    try:
        _add(repository, "a", started, started)
        repository.record_pair_filter(pair, "1.1", eligible=True, now=started)
        repository.rebuild_series(15)
        repository.set_meta("pair_filter_version", "pdop-equals-1.1-v1")
        before = repository.connection.execute(
            "SELECT eligible, series_id FROM observations WHERE image_path=?",
            (pair.image.path,),
        ).fetchone()
        assert before["eligible"] == 1
        assert before["series_id"] is not None

        filter_version = "sign-in-1.01-1.01.5-1.01.6-v2"
        assert repository.ensure_pair_filter_version(filter_version)
        assert repository.connection.execute(
            "SELECT COUNT(*) FROM pair_filters"
        ).fetchone()[0] == 0
        observation = repository.connection.execute(
            "SELECT eligible, series_id FROM observations WHERE image_path=?",
            (pair.image.path,),
        ).fetchone()
        assert observation["eligible"] == 0
        assert observation["series_id"] is None
        assert not repository.ensure_pair_filter_version(filter_version)
    finally:
        repository.close()


def test_series_use_gap_between_consecutive_photos(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    start = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        _add(repository, "a", start, start)
        _add(repository, "b", start + timedelta(minutes=14), start)
        _add(repository, "c", start + timedelta(minutes=30), start)

        repository.rebuild_series(15)
        rows = repository.connection.execute(
            "SELECT stem, series_id FROM observations ORDER BY captured_at"
        ).fetchall()

        assert rows[0]["series_id"] == rows[1]["series_id"]
        assert rows[1]["series_id"] != rows[2]["series_id"]
    finally:
        repository.close()


def test_series_uses_bounded_total_span(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    start = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        _add(repository, "late", start + timedelta(minutes=100), start)
        _add(repository, "first", start, start, place="address A")
        _add(
            repository,
            "middle",
            start + timedelta(minutes=50),
            start,
            place="address B",
        )

        repository.rebuild_series(60)
        rows = repository.connection.execute(
            """
            SELECT stem, series_id
            FROM observations ORDER BY captured_at, source_key
            """
        ).fetchall()

        assert rows[0]["series_id"] == rows[1]["series_id"]
        assert rows[1]["series_id"] != rows[2]["series_id"]
    finally:
        repository.close()


def test_different_equipment_serial_splits_series(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    start = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        _add(repository, "a", start, start, equipment_serial="serial-a")
        _add(repository, "b", start, start, equipment_serial="serial-b")

        repository.rebuild_series(60)
        rows = repository.connection.execute(
            "SELECT series_id FROM observations ORDER BY source_key"
        ).fetchall()

        assert rows[0]["series_id"] != rows[1]["series_id"]
    finally:
        repository.close()


def test_assessment_history_is_append_only_and_retry_keeps_uuid(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        observation_id = _add(repository, "a", started, started)
        repository.rebuild_series(60)
        assessment_id, inference_started = repository.begin_assessment(
            observation_id, "criteria-v1", started
        )
        repository.record_failure(
            observation_id,
            "criteria-v1",
            "timeout",
            max_attempts=3,
            retry_base_seconds=1,
            now=started,
            allow_exhaustion=False,
        )
        retried_id, retried_started = repository.begin_assessment(
            observation_id, "criteria-v1", started + timedelta(seconds=2)
        )

        assert retried_id == assessment_id
        assert retried_started == inference_started

        rich = Assessment(
            82,
            [{"id": "L01", "category": "lawn", "probability": 90}],
            "ok",
            '{"schema_version":2}',
            lawn_probability=90,
            evidence_quality_probability=70,
            target_identity_probability=99,
        )
        saved_id = repository.save_assessment(
            observation_id,
            "criteria-v1",
            rich,
            started + timedelta(seconds=3),
            assessment_id=assessment_id,
            started_at=inference_started,
            model_name="test-model",
            model_parameters={"temperature": 0},
        )
        repository.set_latest_assessment_best(observation_id, True)
        event = repository.connection.execute(
            "SELECT * FROM assessment_events WHERE assessment_id=?",
            (assessment_id,),
        ).fetchone()

        assert saved_id == assessment_id
        assert event["send_probability"] == 82
        assert event["lawn_probability"] == 90
        assert event["model_name"] == "test-model"
        assert event["best"] == 0
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            repository.connection.execute(
                "UPDATE assessment_events SET best=1 WHERE assessment_id=?",
                (assessment_id,),
            )
    finally:
        repository.close()


def test_cursor_export_is_stable_and_contains_snapshots(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        first = _add(repository, "a", started, started)
        second = _add(repository, "b", started + timedelta(minutes=1), started)
        repository.save_assessment(first, "criteria-v1", _assessment(40), started)
        repository.save_assessment(second, "criteria-v1", _assessment(80), started)

        first_page, cursor = repository.export_assessments(0, limit=1)
        repeated, repeated_cursor = repository.export_assessments(0, limit=1)
        second_page, final_cursor = repository.export_assessments(cursor, limit=1)

        assert first_page == repeated
        assert cursor == repeated_cursor == first_page[0]["event_id"]
        assert second_page[0]["event_id"] > cursor
        assert final_cursor == second_page[0]["event_id"]
        assert "raw_response" in first_page[0]
        assert "criteria_details" in first_page[0]
        assert repository.export_assessments(final_cursor)[0] == []
    finally:
        repository.close()


def test_best_is_published_only_after_quiet_window(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        first = _add(repository, "a", started, started)
        second = _add(repository, "b", started + timedelta(minutes=5), started)
        repository.rebuild_series(15)
        repository.save_assessment(first, "criteria-v1", _assessment(40), started)
        repository.save_assessment(second, "criteria-v1", _assessment(90), started)

        open_updates = repository.output_updates(
            "criteria-v1", 15, started + timedelta(minutes=14)
        )
        assert all("best=false" in update.content for update in open_updates)
        assert all("schema_version=2" in update.content for update in open_updates)
        assert all(
            "series_contract_version=" not in update.content
            for update in open_updates
        )

        closed_updates = repository.output_updates(
            "criteria-v1", 15, started + timedelta(minutes=16)
        )
        by_path = {update.remote_path: update.content for update in closed_updates}
        assert "best=false" in by_path["/camera/a.txt"]
        assert "best=true" in by_path["/camera/b.txt"]
    finally:
        repository.close()


def test_incomplete_series_cannot_select_best(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        first = _add(repository, "a", started, started)
        _add(repository, "b", started + timedelta(minutes=5), started)
        repository.rebuild_series(15)
        repository.save_assessment(first, "criteria-v1", _assessment(99), started)

        updates = repository.output_updates(
            "criteria-v1", 15, started + timedelta(minutes=16)
        )

        assert len(updates) == 1
        assert "best=false" in updates[0].content
    finally:
        repository.close()


def test_new_jobs_are_ordered_by_capture_time(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    discovered = datetime(2026, 8, 7, tzinfo=UTC)
    try:
        newer = _add(
            repository,
            "newer",
            datetime(2026, 8, 6, tzinfo=UTC),
            discovered - timedelta(minutes=10),
        )
        older = _add(
            repository,
            "older",
            datetime(2026, 7, 1, tzinfo=UTC),
            discovered,
        )

        jobs = repository.next_new_jobs("criteria-v1", discovered, limit=10)

        assert [job.id for job in jobs] == [older, newer]
    finally:
        repository.close()


def test_progress_snapshot_uses_current_criteria(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        assessed = _add(repository, "assessed", started, started)
        _add(repository, "pending", started + timedelta(minutes=1), started)
        repository.save_assessment(
            assessed, "criteria-v1", _assessment(80), started
        )

        progress = repository.record_progress_snapshot(
            "criteria-v1",
            ftp_total_pairs=3,
            ftp_stable_pairs=2,
            now=started,
        )

        assert progress == {
            "ftp_total_pairs": 3,
            "ftp_stable_pairs": 2,
            "discovered_total": 2,
            "assessed_current": 1,
            "awaiting_assessment": 2,
            "failed_current": 0,
        }
        assert not repository.progress_report_due(
            "criteria-v1", 3600, started + timedelta(minutes=59)
        )
        assert repository.progress_report_due(
            "criteria-v1", 3600, started + timedelta(hours=1)
        )
        assert repository.progress_report_due(
            "criteria-v2", 3600, started + timedelta(minutes=1)
        )
    finally:
        repository.close()


def test_transient_failure_is_not_exhausted(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        observation_id = _add(repository, "retry", started, started)

        exhausted = repository.record_failure(
            observation_id,
            "criteria-v1",
            "AI request failed after retries: HTTP 429",
            max_attempts=1,
            retry_base_seconds=30,
            now=started,
            allow_exhaustion=False,
        )
        row = repository.connection.execute(
            "SELECT * FROM observations WHERE id=?", (observation_id,)
        ).fetchone()

        assert not exhausted
        assert row["failed_criteria_hash"] is None
        assert row["retry_after"] is not None
        assert row["needs_new_assessment"] == 1
    finally:
        repository.close()


def test_legacy_transient_failure_is_requeued(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    started = datetime(2026, 8, 1, tzinfo=UTC)
    try:
        observation_id = _add(repository, "legacy", started, started)
        repository.record_failure(
            observation_id,
            "criteria-v1",
            "AI request failed after retries: timeout",
            max_attempts=1,
            retry_base_seconds=30,
            now=started,
        )

        assert repository.release_legacy_transient_ai_failures("criteria-v1") == 1
        row = repository.connection.execute(
            "SELECT * FROM observations WHERE id=?", (observation_id,)
        ).fetchone()
        assert row["failed_criteria_hash"] is None
        assert row["retry_after"] is None
        assert row["attempt_count"] == 0
    finally:
        repository.close()


def test_existing_database_gets_eligible_column(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY,
            series_id TEXT,
            captured_at TEXT NOT NULL,
            needs_new_assessment INTEGER NOT NULL DEFAULT 1,
            criteria_hash TEXT,
            retry_after TEXT
        );
        INSERT INTO observations (id, captured_at)
        VALUES (1, '2026-08-01T00:00:00+00:00');
        """
    )
    connection.close()

    repository = Repository(db_path)
    try:
        columns = {
            row["name"]
            for row in repository.connection.execute(
                "PRAGMA table_info(observations)"
            )
        }
        row = repository.connection.execute(
            "SELECT eligible FROM observations WHERE id=1"
        ).fetchone()

        assert "eligible" in columns
        assert row["eligible"] == 0
    finally:
        repository.close()


def test_daily_assessment_log_uses_timezone_and_tracks_best(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    assessed_at = datetime(2026, 8, 18, 21, 5, 6, tzinfo=UTC)
    try:
        observation_id = _add(repository, "fact-1", assessed_at, assessed_at)
        repository.save_assessment(
            observation_id,
            "criteria-v1",
            _assessment(87),
            assessed_at,
        )
        repository.set_latest_assessment_best(observation_id, True)

        updates = repository.assessment_log_updates(
            timezone(timedelta(hours=3)), assessed_at
        )

        assert len(updates) == 1
        assert updates[0].log_date == "19-08-2026"
        assert updates[0].content == (
            "дата оценки\tвремя оценки\tпапка на ftp сервере\t"
            "имя факта\tоценка\tлучший\n"
            "19-08-2026\t00:05:06\t/camera\tfact-1\t87\ttrue\n"
        )

        repository.mark_assessment_log_published(
            updates[0].log_date, updates[0].content_hash, assessed_at
        )
        assert not repository.assessment_log_updates(
            timezone(timedelta(hours=3)), assessed_at
        )

        repository.set_latest_assessment_best(observation_id, False)
        changed = repository.assessment_log_updates(
            timezone(timedelta(hours=3)), assessed_at
        )
        assert len(changed) == 1
        assert changed[0].content.endswith("\tfalse\n")
    finally:
        repository.close()


def test_daily_log_is_created_without_assessments(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    now = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    try:
        updates = repository.assessment_log_updates(
            timezone(timedelta(hours=3)), now
        )

        assert len(updates) == 1
        assert updates[0].log_date == "19-08-2026"
        assert updates[0].content == (
            "дата оценки\tвремя оценки\tпапка на ftp сервере\t"
            "имя факта\tоценка\tлучший\n"
        )
    finally:
        repository.close()


def test_preexisting_assessment_events_are_marked_legacy(tmp_path) -> None:
    db_path = tmp_path / "legacy-events.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY,
            series_id TEXT,
            captured_at TEXT NOT NULL,
            needs_new_assessment INTEGER NOT NULL DEFAULT 1,
            criteria_hash TEXT,
            retry_after TEXT
        );
        INSERT INTO observations (id, captured_at)
        VALUES (1, '2026-08-09T10:00:00+00:00');

        CREATE TABLE assessment_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_id INTEGER NOT NULL,
            directory TEXT NOT NULL,
            stem TEXT NOT NULL,
            assessed_at TEXT NOT NULL,
            probability INTEGER NOT NULL,
            criteria_hash TEXT NOT NULL,
            best INTEGER NOT NULL DEFAULT 0,
            UNIQUE(observation_id, assessed_at, criteria_hash)
        );
        INSERT INTO assessment_events (
            observation_id, directory, stem, assessed_at,
            probability, criteria_hash, best
        ) VALUES (
            1, '/camera', 'old-fact', '2026-08-09T10:00:00+00:00',
            75, 'criteria-v1', 0
        );
        """
    )
    connection.close()

    repository = Repository(db_path)
    now = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
    try:
        source = repository.connection.execute(
            "SELECT source FROM assessment_events"
        ).fetchone()["source"]
        updates = repository.assessment_log_updates(
            timezone(timedelta(hours=3)), now
        )

        assert source == "legacy"
        assert len(updates) == 1
        assert updates[0].log_date == "19-08-2026"
        assert "old-fact" not in updates[0].content
    finally:
        repository.close()

    # Simulate a rollback to the old worker, which can still append a row using
    # only its legacy columns, then move forward to the aligned worker again.
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO assessment_events (
            observation_id, directory, stem, assessed_at,
            probability, criteria_hash, best
        ) VALUES (1, '/camera', 'rollback-fact',
                  '2026-08-09T11:00:00+00:00', 65, 'criteria-v1', 0)
        """
    )
    connection.commit()
    connection.close()

    reopened = Repository(db_path)
    try:
        migrated = reopened.connection.execute(
            "SELECT send_probability FROM assessment_events WHERE stem='rollback-fact'"
        ).fetchone()
        assert migrated["send_probability"] == 65
    finally:
        reopened.close()


def test_previous_criteria_version_can_be_reactivated_for_rollback(tmp_path) -> None:
    repository = Repository(tmp_path / "state.db")
    first = CriteriaSet(("first",), "sha256:" + "1" * 64, version="v1")
    second = CriteriaSet(("second",), "sha256:" + "2" * 64, version="v2")
    try:
        repository.register_criteria(first, activated=True)
        repository.register_criteria(second, activated=True)
        repository.register_criteria(first, activated=True)

        active = repository.active_criteria()
        rows = repository.connection.execute(
            "SELECT version, retired_at FROM criteria_versions ORDER BY version"
        ).fetchall()
        assert active is not None and active["version"] == "v1"
        assert rows[0]["retired_at"] is None
        assert rows[1]["retired_at"] is not None
    finally:
        repository.close()
