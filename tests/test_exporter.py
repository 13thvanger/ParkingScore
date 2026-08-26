from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import pytest

from parking_score.__main__ import main
from parking_score.database import Repository
from parking_score.models import Assessment, PhotoMetadata, RemoteFile, RemotePair


def test_export_cli_uses_cursor_without_service_secrets(
    tmp_path, monkeypatch, capsys
) -> None:
    database = tmp_path / "state.db"
    repository = Repository(database)
    pair = RemotePair(
        RemoteFile("/root/fact.jpg", 100, "one"),
        RemoteFile("/root/fact.xml", 200, "one"),
    )
    metadata = PhotoMetadata(
        capture_id="capture",
        plate="A001AA48",
        place="test",
        camera="serial",
        equipment_serial="serial",
        captured_at=datetime(2026, 8, 1, tzinfo=UTC),
        image_width=None,
        image_height=None,
        plate_box=None,
        group_key="A001AA48\x1fserial",
        sign="1.01",
        canonical_sign="1.01",
    )
    observation_id, _ = repository.upsert_observation(
        pair, metadata, tmp_path / "cache.jpg"
    )
    repository.rebuild_series(60)
    assessment_id = repository.save_assessment(
        observation_id,
        "sha256:" + "1" * 64,
        Assessment(67, [], "comment", "raw"),
        model_name="model",
    )
    repository.close()
    output = tmp_path / "assessments.ndjson"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parking-score",
            "export",
            "--state-db",
            str(database),
            "--after-id",
            "0",
            "--output",
            str(output),
        ],
    )

    main()

    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["assessment_id"] == assessment_id
    assert exported["event_id"] == 1
    assert "next_after_id=1" in capsys.readouterr().out


def test_evaluate_requires_no_publish_before_loading_environment(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parking-score",
            "evaluate",
            "--criteria",
            str(tmp_path / "candidate.txt"),
            "--dataset",
            str(tmp_path / "dataset.ndjson"),
            "--output",
            str(tmp_path / "result.ndjson"),
            "--env-file",
            str(tmp_path / "does-not-exist.env"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 2
    assert "requires explicit --no-publish" in capsys.readouterr().err
