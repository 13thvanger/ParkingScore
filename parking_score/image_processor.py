from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .models import Observation


class ImagePreparationError(ValueError):
    """Raised when an image cannot be prepared for the AI API."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    data_url: str
    width: int
    height: int
    byte_size: int


def _mark_target(image: Image.Image, observation: Observation) -> None:
    box = observation.plate_box
    if box is None:
        return
    source_width = observation.image_width or image.width
    source_height = observation.image_height or image.height
    if source_width <= 0 or source_height <= 0:
        return
    scale_x = image.width / source_width
    scale_y = image.height / source_height
    x1 = round(box.x1 * scale_x)
    y1 = round(box.y1 * scale_y)
    x2 = round(box.x2 * scale_x)
    y2 = round(box.y2 * scale_y)
    margin = max(8, round(min(image.size) * 0.008))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.width - 1, x2 + margin)
    y2 = min(image.height - 1, y2 + margin)
    width = max(4, round(min(image.size) * 0.004))

    draw = ImageDraw.Draw(image)
    color = (255, 0, 255)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    label = "TARGET"
    font = ImageFont.load_default()
    label_box = draw.textbbox((x1, y1), label, font=font)
    label_width = label_box[2] - label_box[0] + 8
    label_height = label_box[3] - label_box[1] + 6
    label_y = max(0, y1 - label_height)
    draw.rectangle(
        (x1, label_y, min(image.width - 1, x1 + label_width), y1), fill=color
    )
    draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)


def prepare_image(
    image_path: str,
    observation: Observation,
    max_dimension: int,
    jpeg_quality: int,
    max_bytes: int,
) -> PreparedImage:
    try:
        with Image.open(image_path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImagePreparationError(f"Cannot decode image: {exc}") from exc

    if max(image.size) > max_dimension:
        ratio = max_dimension / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    _mark_target(image, observation)
    quality = jpeg_quality
    encoded = b""
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= max_bytes:
            break
        if quality > 55:
            quality = max(55, quality - 10)
            continue
        if max(image.size) <= 640:
            raise ImagePreparationError(
                f"Prepared image remains larger than {max_bytes} bytes"
            )
        ratio = max(640 / max(image.size), 0.8)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
        quality = min(jpeg_quality, 75)

    payload = base64.b64encode(encoded).decode("ascii")
    return PreparedImage(
        data_url=f"data:image/jpeg;base64,{payload}",
        width=image.width,
        height=image.height,
        byte_size=len(encoded),
    )
