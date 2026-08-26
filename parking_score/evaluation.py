from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .ai_client import AIClient
from .config import Settings
from .criteria import CriteriaSet
from .database import Repository, utc_now
from .ftp_client import FtpClient
from .image_processor import prepare_image
from .xml_parser import parse_recognition_xml


class EvaluationError(ValueError):
    """Raised when a fixed evaluation dataset cannot be processed safely."""


@dataclass(frozen=True, slots=True)
class DatasetItem:
    source_key: str
    capture_id: str


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    evaluation_run_id: str
    total: int
    completed: int
    failed: int
    skipped: int


def load_dataset(path: Path) -> tuple[list[DatasetItem], str]:
    items: list[DatasetItem] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise EvaluationError(f"Cannot read evaluation dataset {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"Invalid dataset JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvaluationError(f"Dataset line {line_number} must be an object")
        source_key = value.get("source_key")
        capture_id = value.get("capture_id")
        if not isinstance(source_key, str) or not source_key.strip():
            raise EvaluationError(
                f"Dataset line {line_number} has no source_key"
            )
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise EvaluationError(
                f"Dataset line {line_number} has no capture_id"
            )
        source_key = source_key.strip()
        capture_id = capture_id.strip()
        if source_key in seen:
            raise EvaluationError(f"Duplicate dataset source_key: {source_key}")
        seen.add(source_key)
        items.append(DatasetItem(source_key, capture_id))
    if not items:
        raise EvaluationError("Evaluation dataset is empty")
    canonical = json.dumps(
        [
            {"capture_id": item.capture_id, "source_key": item.source_key}
            for item in sorted(items, key=lambda item: item.source_key)
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return items, f"sha256:{digest}"


def write_ndjson_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        repository: Repository | None = None,
        ai_client: AIClient | None = None,
        ftp_factory: type[FtpClient] = FtpClient,
    ) -> None:
        self.settings = settings
        self.settings.ensure_runtime_dirs()
        self.repository = repository or Repository(settings.state_db)
        self.ai_client = ai_client or AIClient(settings)
        self.ftp_factory = ftp_factory

    def close(self) -> None:
        self.ai_client.close()
        self.repository.close()

    def run(
        self, criteria: CriteriaSet, dataset_path: Path, output_path: Path
    ) -> EvaluationSummary:
        items, dataset_hash = load_dataset(dataset_path)
        evaluation_run_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"parking-score:{criteria.content_hash}:{dataset_hash}",
            )
        )
        self.repository.begin_evaluation_run(
            evaluation_run_id, criteria, dataset_hash
        )
        completed = 0
        failed = 0
        skipped = 0
        for item in items:
            assessment_id, status = self.repository.begin_evaluation_item(
                evaluation_run_id, item.source_key, item.capture_id
            )
            if status == "completed":
                completed += 1
                skipped += 1
                continue
            try:
                observation = self.repository.observation_by_source_key(
                    item.source_key
                )
                if observation is None:
                    raise EvaluationError(
                        f"Unknown or ineligible source_key: {item.source_key}"
                    )
                if observation.capture_id != item.capture_id:
                    raise EvaluationError(
                        f"capture_id mismatch for {item.source_key}"
                    )
                started_at = utc_now()
                evaluation_observation = self._prepare_observation(
                    evaluation_run_id, observation
                )
                image_sha256 = hashlib.sha256(
                    evaluation_observation.cache_image_path.read_bytes()
                ).hexdigest()
                prepared = prepare_image(
                    str(evaluation_observation.cache_image_path),
                    evaluation_observation,
                    self.settings.ai_image_max_dimension,
                    self.settings.ai_image_jpeg_quality,
                    self.settings.ai_image_max_bytes,
                )
                assessment = self.ai_client.assess(
                    evaluation_observation, criteria, prepared
                )
                parameters = getattr(
                    self.ai_client,
                    "request_parameters",
                    {
                        "model": self.settings.ai_model,
                        "temperature": self.settings.ai_temperature,
                        "max_tokens": self.settings.ai_max_tokens,
                    },
                )
                self.repository.save_assessment(
                    observation.id,
                    criteria,
                    assessment,
                    assessment_id=assessment_id,
                    started_at=started_at,
                    model_name=self.settings.ai_model,
                    prompt_version=criteria.prompt_version,
                    model_parameters=dict(parameters),
                    mode="shadow",
                    image_sha256=image_sha256,
                    evaluation_run_id=evaluation_run_id,
                )
                self.repository.finish_evaluation_item(
                    evaluation_run_id, item.source_key
                )
                completed += 1
            except Exception as exc:  # noqa: BLE001 - persist per-item failure
                self.repository.finish_evaluation_item(
                    evaluation_run_id, item.source_key, error=str(exc)
                )
                failed += 1
        self.repository.finish_evaluation_run(evaluation_run_id)
        write_ndjson_atomic(
            output_path, self.repository.evaluation_results(evaluation_run_id)
        )
        return EvaluationSummary(
            evaluation_run_id=evaluation_run_id,
            total=len(items),
            completed=completed,
            failed=failed,
            skipped=skipped,
        )

    def _prepare_observation(self, evaluation_run_id, observation):
        suffix = PurePosixPath(observation.image_path).suffix or ".image"
        local_path = (
            self.settings.evaluation_directory
            / "images"
            / evaluation_run_id
            / f"{hashlib.sha256(observation.source_key.encode()).hexdigest()}{suffix}"
        )
        with self.ftp_factory(self.settings) as ftp:
            xml_data = ftp.download_bytes(observation.xml_path)
            metadata = parse_recognition_xml(
                xml_data,
                fallback_camera=str(PurePosixPath(observation.image_path).parent),
            )
            if metadata.capture_id != observation.capture_id:
                raise EvaluationError(
                    f"FTP XML changed for {observation.source_key}"
                )
            if not local_path.exists():
                ftp.download_to(observation.image_path, local_path)
        return replace(
            observation,
            capture_id=metadata.capture_id,
            plate=metadata.plate,
            place=metadata.place,
            camera=metadata.camera,
            equipment_serial=metadata.equipment_serial,
            sign=metadata.sign,
            canonical_sign=metadata.canonical_sign,
            captured_at=metadata.captured_at,
            image_width=metadata.image_width,
            image_height=metadata.image_height,
            plate_box=metadata.plate_box,
            group_key=metadata.group_key,
            cache_image_path=local_path,
        )
