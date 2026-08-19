from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size: int | None
    modified: str | None

    @property
    def signature(self) -> str:
        return f"{self.size if self.size is not None else ''}:{self.modified or ''}"


@dataclass(frozen=True, slots=True)
class RemotePair:
    image: RemoteFile
    xml: RemoteFile

    @property
    def signature(self) -> str:
        return f"image={self.image.signature};xml={self.xml.signature}"


@dataclass(frozen=True, slots=True)
class PlateBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def is_valid(self) -> bool:
        return self.x2 > self.x1 >= 0 and self.y2 > self.y1 >= 0


@dataclass(frozen=True, slots=True)
class PhotoMetadata:
    capture_id: str
    plate: str
    place: str
    camera: str
    captured_at: datetime
    image_width: int | None
    image_height: int | None
    plate_box: PlateBox | None
    group_key: str
    pdop: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    id: int
    directory: str
    stem: str
    image_path: str
    xml_path: str
    pair_signature: str
    plate: str
    place: str
    camera: str
    captured_at: datetime
    discovered_at: datetime
    image_width: int | None
    image_height: int | None
    plate_box: PlateBox | None
    group_key: str
    series_id: str | None
    probability: int | None
    criteria_hash: str | None
    needs_new_assessment: bool
    cache_image_path: Path

    @property
    def output_path(self) -> str:
        if self.directory in ("", "/"):
            return f"/{self.stem}.txt" if self.directory == "/" else f"{self.stem}.txt"
        return f"{self.directory.rstrip('/')}/{self.stem}.txt"


@dataclass(frozen=True, slots=True)
class Assessment:
    probability: int
    criteria_details: list[dict[str, Any]]
    comment: str
    raw_response: str


@dataclass(frozen=True, slots=True)
class OutputUpdate:
    observation_id: int
    remote_path: str
    content: str


@dataclass(frozen=True, slots=True)
class AssessmentLogUpdate:
    log_date: str
    content: str
    content_hash: str
