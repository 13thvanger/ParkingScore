from __future__ import annotations

import io
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Self

from PIL import Image

from parking_score.ai_client import AITransientError
from parking_score.config import Settings
from parking_score.database import to_iso, utc_now
from parking_score.models import Assessment, RemoteFile
from parking_score.service import ParkingScoreService


class FakeFtp:
    def __init__(self, xml: bytes, image: bytes) -> None:
        self.source = {
            "/root/photo-1.xml": xml,
            "/root/photo-1.jpg": image,
        }
        self.uploaded: dict[str, bytes] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def list_files(self, root: str, recursive: bool) -> list[RemoteFile]:
        return [
            RemoteFile(path, len(value), "20260801000000")
            for path, value in self.source.items()
        ]

    def download_bytes(self, path: str) -> bytes:
        return self.source[path]

    def download_to(self, path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.source[path])

    def upload_atomic(self, path: str, content: bytes) -> None:
        self.uploaded[path] = content


class FakeAI:
    def __init__(self) -> None:
        self.calls = 0
        self.stems: list[str] = []

    def assess(self, observation, criteria, image) -> Assessment:
        self.calls += 1
        self.stems.append(observation.stem)
        return Assessment(73, [], "test", '{"probability":73,"criteria":[]}')

    def close(self) -> None:
        return None


class ConcurrentFakeAI(FakeAI):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def assess(self, observation, criteria, image) -> Assessment:
        with self._lock:
            self.calls += 1
            self.stems.append(observation.stem)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.1)
            return Assessment(73, [], "test", '{"probability":73,"criteria":[]}')
        finally:
            with self._lock:
                self._active -= 1


class TransientFakeAI(FakeAI):
    def assess(self, observation, criteria, image) -> Assessment:
        self.calls += 1
        raise AITransientError("temporary AI response")


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 200), "green").save(buffer, "JPEG")
    return buffer.getvalue()


def _xml() -> bytes:
    return b"""<?xml version="1.0" encoding="utf-8"?>
<RecognitionData>
  <CaptureInfo>
    <Id>photo-1</Id><Date>2026-08-01T10:00:00Z</Date><Number>O716MP48</Number>
  </CaptureInfo>
  <ImagesInfo>
    <ImageWidth>320</ImageWidth><ImageHeight>200</ImageHeight>
    <Position><X1>100</X1><Y1>100</Y1><X2>150</X2><Y2>120</Y2></Position>
  </ImagesInfo>
  <Coordinates><Pdop>1.1</Pdop></Coordinates>
  <Address>test address</Address>
  <CameraSerialNumber>camera-1</CameraSerialNumber>
</RecognitionData>"""


def test_cycle_processes_pair_and_finalizes_best(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text("criterion one\n", encoding="utf-8")
    settings = Settings(
        ftp_host="example",
        ftp_port=21,
        ftp_user="user",
        ftp_password="password",
        ftp_root_dir="/root",
        ftp_stable_polls=1,
        ai_api_key="key",
        criteria_file=criteria_path,
        state_db=tmp_path / "state.db",
        cache_dir=tmp_path / "cache",
    )
    fake_ftp = FakeFtp(_xml(), _jpeg())
    fake_ai = FakeAI()
    service = ParkingScoreService(
        settings,
        ai_client=fake_ai,
        ftp_factory=lambda unused: fake_ftp,
    )
    try:
        service.run_cycle()
        assert fake_ai.calls == 1
        assert fake_ftp.uploaded["/root/photo-1.txt"] == (
            b"probability=73\nbest=false\n"
        )
        log_path = next(
            path for path in fake_ftp.uploaded if path.endswith(".log")
        )
        assert log_path.startswith("/")
        assert log_path.count("/") == 1
        log_content = fake_ftp.uploaded[log_path].decode("utf-8")
        assert log_content.startswith(
            "дата оценки\tвремя оценки\tпапка на ftp сервере\t"
            "имя факта\tоценка\tлучший\n"
        )
        assert "\t/root\tphoto-1\t73\tfalse\n" in log_content
        progress = service.repository.connection.execute(
            "SELECT * FROM progress_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert progress["ftp_total_pairs"] == 1
        assert progress["ftp_stable_pairs"] == 1
        assert progress["assessed_current"] == 1
        assert progress["awaiting_assessment"] == 0

        old = utc_now() - timedelta(minutes=16)
        with service.repository.connection:
            service.repository.connection.execute(
                "UPDATE observations SET discovered_at=?", (to_iso(old),)
            )
        service.run_cycle()
        assert fake_ai.calls == 1
        assert fake_ftp.uploaded["/root/photo-1.txt"] == (
            b"probability=73\nbest=true\n"
        )
        assert "\t/root\tphoto-1\t73\ttrue\n" in (
            fake_ftp.uploaded[log_path].decode("utf-8")
        )

        criteria_path.write_text("changed criterion\n", encoding="utf-8")
        fake_ftp.source["/root/photo-2.xml"] = _xml().replace(b"photo-1", b"photo-2")
        fake_ftp.source["/root/photo-2.jpg"] = _jpeg()
        service.run_cycle()
        assert fake_ai.stems == ["photo-1", "photo-2"]

        service.run_cycle()
        assert fake_ai.stems == ["photo-1", "photo-2", "photo-1"]
    finally:
        service.close()


def test_cycle_assesses_images_concurrently(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text("criterion one\n", encoding="utf-8")
    settings = Settings(
        ftp_host="example",
        ftp_port=21,
        ftp_user="user",
        ftp_password="password",
        ftp_root_dir="/root",
        ftp_stable_polls=1,
        ai_api_key="key",
        ai_worker_threads=2,
        criteria_file=criteria_path,
        state_db=tmp_path / "state.db",
        cache_dir=tmp_path / "cache",
    )
    fake_ftp = FakeFtp(_xml(), _jpeg())
    fake_ftp.source["/root/photo-2.xml"] = _xml().replace(
        b"photo-1", b"photo-2"
    )
    fake_ftp.source["/root/photo-2.jpg"] = _jpeg()
    fake_ai = ConcurrentFakeAI()
    service = ParkingScoreService(
        settings,
        ai_client=fake_ai,
        ftp_factory=lambda unused: fake_ftp,
    )
    try:
        service.run_cycle()

        assert fake_ai.calls == 2
        assert fake_ai.max_active == 2
        assert "/root/photo-1.txt" in fake_ftp.uploaded
        assert "/root/photo-2.txt" in fake_ftp.uploaded
    finally:
        service.close()


def test_transient_ai_error_stays_in_queue(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text("criterion one\n", encoding="utf-8")
    settings = Settings(
        ftp_host="example",
        ftp_port=21,
        ftp_user="user",
        ftp_password="password",
        ftp_root_dir="/root",
        ftp_stable_polls=1,
        max_processing_attempts=1,
        ai_api_key="key",
        criteria_file=criteria_path,
        state_db=tmp_path / "state.db",
        cache_dir=tmp_path / "cache",
    )
    fake_ftp = FakeFtp(_xml(), _jpeg())
    service = ParkingScoreService(
        settings,
        ai_client=TransientFakeAI(),
        ftp_factory=lambda unused: fake_ftp,
    )
    try:
        service.run_cycle()
        row = service.repository.connection.execute(
            "SELECT * FROM observations LIMIT 1"
        ).fetchone()

        assert row["failed_criteria_hash"] is None
        assert row["retry_after"] is not None
        assert row["needs_new_assessment"] == 1
    finally:
        service.close()


def test_cycle_excludes_pair_with_other_pdop(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text("criterion one\n", encoding="utf-8")
    settings = Settings(
        ftp_host="example",
        ftp_port=21,
        ftp_user="user",
        ftp_password="password",
        ftp_root_dir="/root",
        ftp_stable_polls=1,
        ai_api_key="key",
        criteria_file=criteria_path,
        state_db=tmp_path / "state.db",
        cache_dir=tmp_path / "cache",
    )
    fake_ftp = FakeFtp(_xml(), _jpeg())
    fake_ftp.source["/root/photo-2.xml"] = (
        _xml()
        .replace(b"photo-1", b"photo-2")
        .replace(b"<Pdop>1.1</Pdop>", b"<Pdop>2.0</Pdop>")
    )
    fake_ftp.source["/root/photo-2.jpg"] = _jpeg()
    fake_ai = FakeAI()
    service = ParkingScoreService(
        settings,
        ai_client=fake_ai,
        ftp_factory=lambda unused: fake_ftp,
    )
    try:
        service.run_cycle()

        assert fake_ai.stems == ["photo-1"]
        assert "/root/photo-1.txt" in fake_ftp.uploaded
        assert "/root/photo-2.txt" not in fake_ftp.uploaded
        filters = service.repository.connection.execute(
            "SELECT image_path, pdop, eligible FROM pair_filters ORDER BY image_path"
        ).fetchall()
        assert [tuple(row) for row in filters] == [
            ("/root/photo-1.jpg", "1.1", 1),
            ("/root/photo-2.jpg", "2.0", 0),
        ]
    finally:
        service.close()


def test_changed_pdop_deactivates_existing_observation(tmp_path) -> None:
    criteria_path = tmp_path / "criteria.txt"
    criteria_path.write_text("criterion one\n", encoding="utf-8")
    settings = Settings(
        ftp_host="example",
        ftp_port=21,
        ftp_user="user",
        ftp_password="password",
        ftp_root_dir="/root",
        ftp_stable_polls=1,
        ai_api_key="key",
        criteria_file=criteria_path,
        state_db=tmp_path / "state.db",
        cache_dir=tmp_path / "cache",
    )
    fake_ftp = FakeFtp(_xml(), _jpeg())
    fake_ai = FakeAI()
    service = ParkingScoreService(
        settings,
        ai_client=fake_ai,
        ftp_factory=lambda unused: fake_ftp,
    )
    try:
        service.run_cycle()
        assert fake_ai.calls == 1

        fake_ftp.source["/root/photo-1.xml"] = _xml().replace(
            b"<Pdop>1.1</Pdop>", b"<Pdop>2.00</Pdop>"
        )
        service.run_cycle()
        row = service.repository.connection.execute(
            "SELECT eligible, series_id FROM observations LIMIT 1"
        ).fetchone()

        assert fake_ai.calls == 1
        assert row["eligible"] == 0
        assert row["series_id"] is None
    finally:
        service.close()
