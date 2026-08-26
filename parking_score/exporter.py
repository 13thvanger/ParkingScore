from __future__ import annotations

import json
import uuid
from pathlib import Path

from .database import Repository


def export_assessments_ndjson(
    repository: Repository,
    output_path: Path,
    *,
    after_id: int = 0,
    limit: int = 500,
) -> tuple[int, int]:
    rows, next_after_id = repository.export_assessments(after_id, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows), next_after_id
