from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path
from typing import Self

from PIL import Image

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

        criteria_path.write_text("changed criterion\n", encoding="utf-8")
        fake_ftp.source["/root/photo-2.xml"] = _xml().replace(b"photo-1", b"photo-2")
        fake_ftp.source["/root/photo-2.jpg"] = _jpeg()
        service.run_cycle()
        assert fake_ai.stems == ["photo-1", "photo-2"]

        service.run_cycle()
        assert fake_ai.stems == ["photo-1", "photo-2", "photo-1"]
    finally:
        service.close()
