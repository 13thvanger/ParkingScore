from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from .models import PhotoMetadata, PlateBox


class MetadataError(ValueError):
    """Raised when a recognition XML does not contain required metadata."""


_CYRILLIC_PLATE_TRANSLATION = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)


def normalize_plate(value: str) -> str:
    value = (
        value.strip().upper().replace("Ё", "Е").translate(_CYRILLIC_PLATE_TRANSLATION)
    )
    return "".join(re.findall(r"[A-Z0-9]", value))


def normalize_text(value: str) -> str:
    return " ".join(value.replace("Ё", "Е").replace("ё", "е").split()).casefold()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _path(root: ET.Element, *names: str) -> ET.Element | None:
    current: ET.Element | None = root
    for name in names:
        if current is None:
            return None
        current = _direct_child(current, name)
    return current


def _text(root: ET.Element, *names: str, required: bool = False) -> str:
    element = _path(root, *names)
    value = (element.text or "").strip() if element is not None else ""
    if required and not value:
        raise MetadataError(f"Missing XML field: {'/'.join(names)}")
    return value


def _optional_int(root: ET.Element, *names: str) -> int | None:
    value = _text(root, *names)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise MetadataError(f"Invalid integer in {'/'.join(names)}: {value}") from exc


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MetadataError(f"Invalid CaptureInfo/Date: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_recognition_xml(data: bytes) -> PhotoMetadata:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise MetadataError(f"Invalid XML: {exc}") from exc

    capture_id = _text(root, "CaptureInfo", "Id", required=True)
    raw_plate = _text(root, "CaptureInfo", "Number", required=True)
    plate = normalize_plate(raw_plate)
    if not plate:
        raise MetadataError("CaptureInfo/Number does not contain a valid plate")

    captured_at = _parse_datetime(_text(root, "CaptureInfo", "Date", required=True))
    place = _text(root, "Address")
    if not place:
        latitude = _text(root, "Coordinates", "Latitude", required=True)
        longitude = _text(root, "Coordinates", "Longitude", required=True)
        try:
            place = f"geo:{float(latitude):.5f},{float(longitude):.5f}"
        except ValueError as exc:
            raise MetadataError("Invalid fallback coordinates") from exc

    camera = _text(root, "CameraSerialNumber", required=True)
    width = _optional_int(root, "ImagesInfo", "ImageWidth")
    height = _optional_int(root, "ImagesInfo", "ImageHeight")

    coords = [
        _optional_int(root, "ImagesInfo", "Position", name)
        for name in ("X1", "Y1", "X2", "Y2")
    ]
    plate_box = None
    if all(value is not None for value in coords):
        candidate = PlateBox(*(int(value) for value in coords if value is not None))
        if candidate.is_valid:
            plate_box = candidate

    group_key = "\x1f".join((plate, normalize_text(place), normalize_text(camera)))
    return PhotoMetadata(
        capture_id=capture_id,
        plate=plate,
        place=place,
        camera=camera,
        captured_at=captured_at,
        image_width=width,
        image_height=height,
        plate_box=plate_box,
        group_key=group_key,
    )
