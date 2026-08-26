from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class CriteriaError(ValueError):
    """Raised when the criteria file cannot be used."""


@dataclass(frozen=True, slots=True)
class Criterion:
    id: str
    category: str
    text: str


@dataclass(frozen=True, slots=True)
class CriteriaSet:
    items: tuple[str, ...]
    content_hash: str
    definitions: tuple[Criterion, ...] = ()
    schema_version: int = 0
    version: str = "legacy"
    prompt_version: str = "parking-lawn-v2"
    source_text: str = ""

    def __post_init__(self) -> None:
        if self.definitions:
            return
        generated = tuple(
            Criterion(f"LEGACY{index:03d}", "decision", text)
            for index, text in enumerate(self.items, start=1)
        )
        object.__setattr__(self, "definitions", generated)

    @property
    def normalized_json(self) -> str:
        return _normalized_json(self.definitions)


_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_V2_CRITERION = re.compile(
    r"^\[(?P<category>lawn|evidence|identity):(?P<id>[A-Za-z0-9._-]+)\]\s+"
    r"(?P<text>.+)$"
)
_GROUP_SECTION = re.compile(r"^\[(?P<category>lawn|evidence|identity)\]$")
_GROUP_INLINE = re.compile(
    r"^\[(?P<category>lawn|evidence|identity)\]\s+(?P<text>.+)$"
)
_CATEGORY_PREFIX = {"lawn": "L", "evidence": "E", "identity": "I"}
PROMPT_VERSION = "parking-lawn-v2"


def _normalized_json(definitions: tuple[Criterion, ...]) -> str:
    return json.dumps(
        [
            {"category": item.category, "id": item.id, "text": item.text}
            for item in definitions
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _criteria_hash(definitions: tuple[Criterion, ...]) -> str:
    digest = hashlib.sha256(
        _normalized_json(definitions).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _automatic_id(category: str, text: str) -> str:
    digest = hashlib.sha256(f"{category}\0{text}".encode()).hexdigest()
    return f"{_CATEGORY_PREFIX[category]}-{digest[:12].upper()}"


def load_criteria(path: Path) -> CriteriaSet:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CriteriaError(f"Cannot read criteria file {path}: {exc}") from exc

    definitions: list[Criterion] = []
    legacy_items: list[str] = []
    current_category: str | None = None
    grouped = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        section = _GROUP_SECTION.fullmatch(line)
        if section:
            current_category = section.group("category")
            grouped = True
            continue
        line = _LIST_PREFIX.sub("", line).strip()
        if not line:
            continue
        explicit = _V2_CRITERION.fullmatch(line)
        inline = _GROUP_INLINE.fullmatch(line)
        if explicit or inline:
            match = explicit or inline
            assert match is not None
            category = match.group("category")
            text = " ".join(match.group("text").split())
            definitions.append(
                Criterion(
                    id=(
                        explicit.group("id")
                        if explicit is not None
                        else _automatic_id(category, text)
                    ),
                    category=category,
                    text=text,
                )
            )
            grouped = True
            continue
        if line.startswith("["):
            raise CriteriaError(f"Unknown criteria group or invalid line: {line}")
        text = " ".join(line.split())
        if current_category is not None:
            definitions.append(
                Criterion(
                    id=_automatic_id(current_category, text),
                    category=current_category,
                    text=text,
                )
            )
            grouped = True
        else:
            legacy_items.append(text)

    if not definitions and not legacy_items:
        raise CriteriaError(f"Criteria file {path} does not contain any criteria")
    if grouped and legacy_items:
        raise CriteriaError("Criteria file cannot mix grouped and legacy criteria")

    if not grouped:
        legacy_definitions = tuple(
            Criterion(f"LEGACY{index:03d}", "decision", text)
            for index, text in enumerate(legacy_items, start=1)
        )
        content_hash = _criteria_hash(legacy_definitions)
        logger.warning(
            "Legacy criteria file loaded without stable IDs path=%s", path
        )
        return CriteriaSet(
            items=tuple(item.text for item in legacy_definitions),
            content_hash=content_hash,
            definitions=legacy_definitions,
            schema_version=0,
            version=f"legacy-{content_hash.removeprefix('sha256:')[:12]}",
            source_text=content,
        )

    definition_tuple = tuple(definitions)
    ids = [item.id for item in definition_tuple]
    if len(ids) != len(set(ids)):
        raise CriteriaError("Criteria IDs must be unique within a version")
    content_hash = _criteria_hash(definition_tuple)
    return CriteriaSet(
        items=tuple(item.text for item in definition_tuple),
        content_hash=content_hash,
        definitions=definition_tuple,
        schema_version=2,
        version=f"txt-{content_hash.removeprefix('sha256:')[:12]}",
        prompt_version=PROMPT_VERSION,
        source_text=content,
    )
