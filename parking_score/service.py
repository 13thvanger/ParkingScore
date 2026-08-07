from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .ai_client import AIClient
from .config import Settings
from .criteria import CriteriaSet, load_criteria
from .database import Repository, to_iso, utc_now
from .ftp_client import FtpClient, build_pairs
from .image_processor import prepare_image
from .models import Observation
from .xml_parser import parse_recognition_xml

logger = logging.getLogger(__name__)


class ParkingScoreService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository | None = None,
        ai_client: AIClient | None = None,
        ftp_factory: Callable[[Settings], FtpClient] = FtpClient,
    ) -> None:
        self.settings = settings
        self.settings.ensure_runtime_dirs()
        self.repository = repository or Repository(settings.state_db)
        self.ai_client = ai_client or AIClient(settings)
        self.ftp_factory = ftp_factory

    def close(self) -> None:
        self.ai_client.close()
        self.repository.close()

    def run_cycle(self) -> None:
        cycle_started = utc_now()
        self._heartbeat(cycle_started)
        criteria = load_criteria(self.settings.criteria_file)
        previous_hash = self.repository.get_meta("criteria_hash")
        if previous_hash and previous_hash != criteria.content_hash:
            logger.info(
                "Criteria changed; existing assessments queued for background refresh"
            )
        self.repository.set_meta("criteria_hash", criteria.content_hash)

        discovered = self._discover()
        self.repository.rebuild_series(self.settings.series_window_minutes)
        processed, mode, published = self._process_jobs(criteria)
        published += self._publish_outputs(criteria)
        self._heartbeat()

        statistics = self.repository.statistics(criteria.content_hash)
        logger.info(
            "Cycle complete discovered=%d processed=%d mode=%s published=%d "
            "total=%d unassessed=%d stale=%d failed=%d",
            discovered,
            processed,
            mode,
            published,
            statistics["total"],
            statistics["unassessed"],
            statistics["stale"],
            statistics["failed"],
        )

    def _discover(self) -> int:
        discovered = 0
        with self.ftp_factory(self.settings) as ftp:
            logger.info(
                "FTP scan started root=%s recursive=%s",
                self.settings.ftp_root_dir,
                self.settings.ftp_recursive,
            )
            files = ftp.list_files(
                self.settings.ftp_root_dir, self.settings.ftp_recursive
            )
            logger.info("FTP listing complete files=%d", len(files))
            stable_counts = self.repository.update_remote_files(files)
            stable_files = [
                item
                for item in files
                if stable_counts.get(item.path, 0) >= self.settings.ftp_stable_polls
            ]
            pairs = build_pairs(stable_files, self.settings.image_extensions)
            logger.info("FTP stable pairs found=%d", len(pairs))
            for index, pair in enumerate(pairs, start=1):
                if not self.repository.pair_needs_ingest(pair):
                    continue
                try:
                    xml_data = ftp.download_bytes(pair.xml.path)
                    metadata = parse_recognition_xml(
                        xml_data,
                        fallback_camera=str(PurePosixPath(pair.image.path).parent),
                    )
                    local_path = self._cache_path(pair.image.path)
                    local_path.unlink(missing_ok=True)
                    _, changed = self.repository.upsert_observation(
                        pair, metadata, local_path
                    )
                    if changed:
                        discovered += 1
                except Exception:
                    logger.exception(
                        "Cannot ingest FTP pair image=%s xml=%s",
                        pair.image.path,
                        pair.xml.path,
                    )
                if index % 500 == 0:
                    logger.info(
                        "FTP metadata progress checked=%d/%d discovered=%d",
                        index,
                        len(pairs),
                        discovered,
                    )
        return discovered

    def _process_jobs(self, criteria: CriteriaSet) -> tuple[int, str, int]:
        now = utc_now()
        if self.repository.has_pending_new(criteria.content_hash):
            jobs = self.repository.next_new_jobs(
                criteria.content_hash,
                now,
                self.settings.max_new_jobs_per_cycle,
            )
            mode = "new"
        else:
            jobs = self.repository.next_stale_jobs(
                criteria.content_hash,
                now,
                self.settings.max_reprocess_jobs_per_cycle,
            )
            mode = "reprocess" if jobs else "idle"

        processed = 0
        published = 0
        for observation in jobs:
            try:
                self._ensure_cached_image(observation)
                prepared = prepare_image(
                    str(observation.cache_image_path),
                    observation,
                    self.settings.ai_image_max_dimension,
                    self.settings.ai_image_jpeg_quality,
                    self.settings.ai_image_max_bytes,
                )
                assessment = self.ai_client.assess(observation, criteria, prepared)
                self.repository.save_assessment(
                    observation.id, criteria.content_hash, assessment
                )
                processed += 1
                published += self._publish_outputs(criteria)
                logger.info(
                    "Image assessed path=%s probability=%d mode=%s",
                    observation.image_path,
                    assessment.probability,
                    mode,
                )
            except Exception as exc:
                exhausted = self.repository.record_failure(
                    observation.id,
                    criteria.content_hash,
                    str(exc),
                    self.settings.max_processing_attempts,
                    self.settings.retry_base_seconds,
                )
                logger.exception(
                    "Image assessment failed path=%s exhausted=%s",
                    observation.image_path,
                    exhausted,
                )
            finally:
                self._heartbeat()
        return processed, mode, published

    def _ensure_cached_image(self, observation: Observation) -> None:
        if observation.cache_image_path.exists():
            return
        with self.ftp_factory(self.settings) as ftp:
            ftp.download_to(observation.image_path, observation.cache_image_path)

    def _publish_outputs(self, criteria: CriteriaSet) -> int:
        updates = self.repository.output_updates(
            criteria.content_hash, self.settings.series_window_minutes
        )
        if not updates:
            return 0
        published = 0
        with self.ftp_factory(self.settings) as ftp:
            for update in updates:
                try:
                    ftp.upload_atomic(
                        update.remote_path, update.content.encode("utf-8")
                    )
                    self.repository.mark_published(
                        update.observation_id, update.content
                    )
                    published += 1
                    logger.info("Result published path=%s", update.remote_path)
                except Exception:
                    logger.exception(
                        "Cannot publish result path=%s", update.remote_path
                    )
        return published

    def _cache_path(self, remote_path: str) -> Path:
        path = PurePosixPath(remote_path)
        safe_parts = [part for part in path.parts if part not in {"", "/"}]
        if any(part in {".", ".."} for part in safe_parts):
            raise ValueError(f"Unsafe remote path: {remote_path}")
        return self.settings.cache_dir.joinpath(*safe_parts)

    def _heartbeat(self, now: datetime | None = None) -> None:
        self.repository.set_meta("heartbeat", to_iso(now or utc_now()))


def run_forever(settings: Settings) -> None:
    stop_event = threading.Event()

    def stop(signum: int, frame: object) -> None:
        logger.info("Received signal %d; stopping after current operation", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    service = ParkingScoreService(settings)
    try:
        while not stop_event.is_set():
            try:
                service.run_cycle()
            except Exception:
                logger.exception("Service cycle failed")
                service._heartbeat()
            stop_event.wait(settings.poll_interval_seconds)
    finally:
        service.close()


def healthcheck(settings: Settings) -> bool:
    if not settings.state_db.exists():
        return False
    repository = Repository(settings.state_db)
    try:
        heartbeat = repository.get_meta("heartbeat")
    finally:
        repository.close()
    if heartbeat is None:
        return False
    value = datetime.fromisoformat(heartbeat)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    age = datetime.now(UTC) - value.astimezone(UTC)
    return age.total_seconds() <= settings.healthcheck_max_age_seconds
