from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from PIL import Image

from parking_score.config import Settings
from parking_score.criteria import CriteriaSet, Criterion
from parking_score.database import Repository
from parking_score.evaluation import EvaluationRunner
from parking_score.models import Assessment, PhotoMetadata, RemoteFile, RemotePair


class FakeAI:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def request_parameters(self):
        return {"model": "shadow-model", "temperature": 0.0}

    def assess(self, observation, criteria, image) -> Assessment:
        self.calls += 1
        return Assessment(
            88,
            [{"id": "L01", "category": "lawn", "probability": 90}],
            "shadow",
            "{}",
            lawn_probability=90,
            evidence_quality_probability=80,
            target_identity_probability=99,
        )

    def close(self) -> None:
        return None


class FakeFtp:
    def __init__(self, xml: bytes, image: bytes) -> None:
        self.xml = xml
        self.image = image
        self.upload_calls = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def download_bytes(self, path: str) -> bytes:
        return self.xml

    def download_to(self, path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.image)

    def upload_atomic(self, path: str, content: bytes) -> None:
        self.upload_calls += 1


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 200), "green").save(buffer, "JPEG")
    return buffer.getvalue()


def _xml() -> bytes:
    return b"""<RecognitionData>
<SerialNumber>equipment-1</SerialNumber>
<CaptureInfo><Id>capture-1</Id><Date>2026-08-01T10:00:00Z</Date>
<Number>O716MP48</Number></CaptureInfo>
<ImagesInfo><ImageWidth>320</ImageWidth><ImageHeight>200</ImageHeight></ImagesInfo>
<Address>test</Address><Sign>1.01</Sign></RecognitionData>"""


def _criteria() -> CriteriaSet:
    criterion = Criterion("L01", "lawn", "all wheels outside asphalt")
    return CriteriaSet(
        items=(criterion.text,),
        content_hash="sha256:" + "2" * 64,
        definitions=(criterion,),
        schema_version=2,
        version="candidate-1",
    )


def test_evaluate_is_resumable_and_does_not_change_live_state(tmp_path) -> None:
    settings = Settings(
        ftp_host="example",
        ftp_port=21,
        ftp_user="user",
        ftp_password="not-real",
        ai_api_key="not-real",
        state_db=tmp_path / "state.db",
        cache_dir=tmp_path / "cache",
        evaluation_directory=tmp_path / "evaluation",
    )
    repository = Repository(settings.state_db)
    pair = RemotePair(
        RemoteFile("/root/fact-1.jpg", 100, "one"),
        RemoteFile("/root/fact-1.xml", 200, "one"),
    )
    metadata = PhotoMetadata(
        capture_id="capture-1",
        plate="O716MP48",
        place="test",
        camera="equipment-1",
        equipment_serial="equipment-1",
        captured_at=datetime(2026, 8, 1, 10, tzinfo=UTC),
        image_width=320,
        image_height=200,
        plate_box=None,
        group_key="O716MP48\x1fequipment-1",
        sign="1.01",
        canonical_sign="1.01",
    )
    observation_id, _ = repository.upsert_observation(
        pair, metadata, tmp_path / "live-cache.jpg"
    )
    repository.rebuild_series(60)
    live_assessment_id = repository.save_assessment(
        observation_id,
        "sha256:" + "0" * 64,
        Assessment(10, [], "live", "{}"),
        model_name="live-model",
    )
    dataset = tmp_path / "facts.ndjson"
    dataset.write_text(
        json.dumps(
            {"source_key": "/root/fact-1", "capture_id": "capture-1"}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.ndjson"
    fake_ai = FakeAI()
    fake_ftp = FakeFtp(_xml(), _jpeg())
    runner = EvaluationRunner(
        settings,
        repository=repository,
        ai_client=fake_ai,
        ftp_factory=lambda unused: fake_ftp,
    )
    try:
        first = runner.run(_criteria(), dataset, output)
        second = runner.run(_criteria(), dataset, output)

        assert first.completed == 1 and first.failed == 0
        assert second.skipped == 1
        assert fake_ai.calls == 1
        assert fake_ftp.upload_calls == 0
        live = repository.connection.execute(
            "SELECT assessment_id, send_probability FROM observations WHERE id=?",
            (observation_id,),
        ).fetchone()
        assert live["assessment_id"] == live_assessment_id
        assert live["send_probability"] == 10
        shadow = repository.connection.execute(
            "SELECT * FROM assessment_events WHERE mode='shadow'"
        ).fetchall()
        assert len(shadow) == 1
        assert shadow[0]["image_sha256"] == hashlib.sha256(_jpeg()).hexdigest()
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["status"] == "completed"
        assert result["assessment"]["send_probability"] == 88
    finally:
        runner.close()
