"""One-shot plate OCR experiment. Local inputs/output; never publishes to FTP."""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import re
import uuid
from dataclasses import replace
from pathlib import Path

from PIL import Image

from .ai_client import AIClient, AIError, _extract_json_object, _message_text
from .config import Settings
from .xml_parser import normalize_plate, parse_recognition_xml


def prepare_plate_crop(image_path: Path, metadata) -> str:
    box = metadata.plate_box
    if box is None or not box.is_valid:
        raise ValueError("Missing or invalid plate box")
    with Image.open(image_path) as source:
        source.load()
        if source.getexif().get(274, 1) != 1:
            raise ValueError("EXIF rotation requires explicit XML coordinate verification")
        width, height = metadata.image_width, metadata.image_height
        if not width or not height or width <= 0 or height <= 0:
            raise ValueError("XML image dimensions are required")
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise ValueError("Plate box outside XML image dimensions")
        sx, sy = source.width / width, source.height / height
        if not math.isclose(sx, sy, rel_tol=0.01):
            raise ValueError("Image aspect ratio differs from XML")
        # A small border preserves plate edges without adding the whole vehicle.
        mx, my = (box.x2 - box.x1) * 0.1, (box.y2 - box.y1) * 0.1
        crop = source.crop((
            max(0, math.floor((box.x1 - mx) * sx)),
            max(0, math.floor((box.y1 - my) * sy)),
            min(source.width, math.ceil((box.x2 + mx) * sx)),
            min(source.height, math.ceil((box.y2 + my) * sy)),
        )).convert("RGB")
        crop.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def ocr_payload(settings: Settings, image_url: str) -> dict:
    return {
        "model": settings.ai_model,
        "stream": False,
        "temperature": settings.ai_temperature,
        "max_tokens": settings.ai_max_tokens,
        "messages": [
            {"role": "system", "content": (
                "Ты распознаёшь государственные номера автомобилей по изображению. "
                "Не угадывай невидимые символы. Не выводи рассуждения. Только JSON."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "Прочитай единственный номер на фрагменте, включая регион. "
                    "Если есть несколько номеров, не виден регион или хотя бы один "
                    "символ неоднозначен, верни readable=false, plate=null. "
                    "Не дополняй скрытые символы по шаблону. "
                    'Формат: {"plate":"А123ВС48","readable":true,"confidence":95}. '
                    "Пример номера в формате вымышленный; не копируй его. "
                    "confidence — число 0..100, readable — boolean."
                )},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    }


def compare_ocr(content: str, expected: str, threshold: int) -> dict:
    value = _extract_json_object(content)
    readable, confidence, plate = (
        value.get("readable"), value.get("confidence"), value.get("plate")
    )
    if type(readable) is not bool:
        raise AIError("OCR readable must be boolean")
    if type(confidence) not in (int, float) or not 0 <= confidence <= 100:
        raise AIError("OCR confidence must be a number in 0..100")
    if plate is not None and not isinstance(plate, str):
        raise AIError("OCR plate must be text or null")
    # Trial supports ordinary Russian car plates only. Do not silently drop punctuation.
    valid_chars = isinstance(plate, str) and re.fullmatch(r"[АВЕКМНОРСТУХABEKMHOPCTYX0-9\s]+", plate.upper())
    recognized = normalize_plate(plate) if valid_chars else ""
    supported = bool(re.fullmatch(r"[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}", recognized))
    status = "uncertain"
    if readable and supported and confidence >= threshold:
        status = "match" if recognized == normalize_plate(expected) else "mismatch"
    return {
        "plate_check_status": status,
        "plate_xml": expected,
        "plate_recognized": recognized if readable and supported else None,
        "plate_recognition_confidence": confidence,
        "confidence_threshold": threshold,
    }


def run_trial(settings: Settings, image: Path, xml: Path, threshold: int = 90,
              transport=None) -> dict:
    if not 0 <= threshold <= 100:
        raise ValueError("Confidence threshold must be 0..100")
    metadata = parse_recognition_xml(xml.read_bytes(), fallback_camera="ocr-trial")
    payload = ocr_payload(settings, prepare_plate_crop(image, metadata))
    request_id = uuid.uuid4().hex
    client = AIClient(replace(settings, ai_debug_export_enabled=False, ai_request_retries=1), transport)
    try:
        client._wait_for_request_slot()
        response = client._post_with_diagnostics(payload, request_id, request_id, 1)
        if response.status_code != 200:
            raise AIError(f"OCR HTTP status={response.status_code} request_id={request_id}")
        body = response.json()
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if choices and isinstance(choices[0], dict) and choices[0].get("finish_reason") == "length":
            raise AIError(f"OCR token limit exhausted request_id={request_id}")
        result = compare_ocr(_message_text(body), metadata.plate, threshold)
        result.update(request_id=request_id, model=settings.ai_model, mode="ocr-trial")
        return result
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--confidence-threshold", type=int, default=90)
    parser.add_argument("--no-publish", action="store_true", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Reserve a new output before spending an AI call; never overwrite an input/result.
    with args.output.open("x", encoding="utf-8") as output:
        try:
            result = run_trial(Settings.from_env(args.env_file), args.image, args.xml,
                               args.confidence_threshold)
        except Exception as exc:  # noqa: BLE001 - save a safe local failure report
            json.dump({"plate_check_status": "error", "error_type": type(exc).__name__}, output)
            raise SystemExit("OCR trial failed; see diagnostics and local report") from None
        json.dump(result, output, ensure_ascii=False, indent=2)
    print("OCR trial finished; result saved locally")


if __name__ == "__main__":
    main()
