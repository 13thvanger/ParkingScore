from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


class CriteriaError(ValueError):
    """Raised when the criteria file cannot be used."""


@dataclass(frozen=True, slots=True)
class CriteriaSet:
    items: tuple[str, ...]
    content_hash: str


_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


def load_criteria(path: Path) -> CriteriaSet:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CriteriaError(f"Cannot read criteria file {path}: {exc}") from exc

    items: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _LIST_PREFIX.sub("", line).strip()
        if line:
            items.append(line)
    if not items:
        raise CriteriaError(f"Criteria file {path} does not contain any criteria")

    canonical = "\n".join(items)
    return CriteriaSet(
        items=tuple(items),
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
