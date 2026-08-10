from datetime import UTC, datetime, timedelta
from pathlib import Path

from parking_score.database import Repository
from parking_score.models import Assessment, PhotoMetadata, RemoteFile, RemotePair


def _add(
    repository: Repository,
    name: str,
    captured_at: datetime,
    discovered_at: datetime,
) -> int:
    pair = RemotePair(
        RemoteFile(f"/camera/{name}.jpg", 100, "20260801000000"),
        RemoteFile(f"/camera/{name}.xml", 50, "20260801000000"),
    )
    metadata = PhotoMetadata(
        capture_id=name,
        plate="O716MP48",
        place="test address",
        camera="camera-1",
        captured_at=captured_at,
        image_width=1920,
        image_height=1200,
        plate_box=None,
        group_key="O716MP48\x1ftest address\x1fcamera-1",
    )
    observation_id, _ = repository.upsert_observation(
        pair, metadata, Path(f"/tmp/{name}.jpg"), now=discovered_at
    )
    return observation_id


def _assessment(probability: int) -> Assessment:
    return Assessment(probability, [], "", "{}")


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
