from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required service configuration is invalid."""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Environment variable {name} is required")
    return value


def _integer(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


def _floating(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class Settings:
    ftp_host: str
    ftp_port: int
    ftp_user: str
    ftp_password: str = field(repr=False)
    ftp_root_dir: str = "/"
    ftp_recursive: bool = True
    ftp_passive: bool = True
    ftp_timeout_seconds: float = 30.0
    ftp_encoding: str = "utf-8"
    ftp_stable_polls: int = 2

    poll_interval_seconds: int = 60
    series_window_minutes: int = 15
    max_new_jobs_per_cycle: int = 20
    max_reprocess_jobs_per_cycle: int = 1
    max_processing_attempts: int = 5
    retry_base_seconds: int = 30

    ai_api_url: str = "https://bridge-back.admlr.lipetsk.ru/api/v1/chat/completions"
    ai_api_key: str = field(default="", repr=False)
    ai_model: str = "cifra48/agent"
    ai_timeout_seconds: float = 120.0
    ai_request_retries: int = 3
    ai_temperature: float = 0.0
    ai_max_tokens: int = 1000
    ai_image_max_dimension: int = 1920
    ai_image_jpeg_quality: int = 88
    ai_image_max_bytes: int = 5_000_000

    criteria_file: Path = Path("criteria.txt")
    state_db: Path = Path("data/parking_score.db")
    cache_dir: Path = Path("data/cache")
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    log_level: str = "INFO"
    healthcheck_max_age_seconds: int = 600

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file:
            load_dotenv(dotenv_path=env_file, override=False)

        extensions = tuple(
            item if item.startswith(".") else f".{item}"
            for item in (
                part.strip().lower()
                for part in os.getenv(
                    "IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp"
                ).split(",")
            )
            if item
        )
        if not extensions:
            raise ConfigurationError("IMAGE_EXTENSIONS cannot be empty")

        quality = _integer("AI_IMAGE_JPEG_QUALITY", 88)
        if quality > 95:
            raise ConfigurationError("AI_IMAGE_JPEG_QUALITY must be <= 95")

        return cls(
            ftp_host=_required("FTP_HOST"),
            ftp_port=_integer("FTP_PORT", 21),
            ftp_user=_required("FTP_USER"),
            ftp_password=_required("FTP_PASSWORD"),
            ftp_root_dir=os.getenv("FTP_ROOT_DIR", "/").strip() or "/",
            ftp_recursive=_boolean("FTP_RECURSIVE", True),
            ftp_passive=_boolean("FTP_PASSIVE", True),
            ftp_timeout_seconds=_floating("FTP_TIMEOUT_SECONDS", 30.0, 1.0),
            ftp_encoding=os.getenv("FTP_ENCODING", "utf-8").strip() or "utf-8",
            ftp_stable_polls=_integer("FTP_STABLE_POLLS", 2),
            poll_interval_seconds=_integer("POLL_INTERVAL_SECONDS", 60),
            series_window_minutes=_integer("SERIES_WINDOW_MINUTES", 15),
            max_new_jobs_per_cycle=_integer("MAX_NEW_JOBS_PER_CYCLE", 20),
            max_reprocess_jobs_per_cycle=_integer("MAX_REPROCESS_JOBS_PER_CYCLE", 1),
            max_processing_attempts=_integer("MAX_PROCESSING_ATTEMPTS", 5),
            retry_base_seconds=_integer("RETRY_BASE_SECONDS", 30),
            ai_api_url=os.getenv(
                "AI_API_URL",
                "https://bridge-back.admlr.lipetsk.ru/api/v1/chat/completions",
            ).strip(),
            ai_api_key=_required("AI_API_KEY"),
            ai_model=os.getenv("AI_MODEL", "cifra48/agent").strip() or "cifra48/agent",
            ai_timeout_seconds=_floating("AI_TIMEOUT_SECONDS", 120.0, 1.0),
            ai_request_retries=_integer("AI_REQUEST_RETRIES", 3),
            ai_temperature=_floating("AI_TEMPERATURE", 0.0),
            ai_max_tokens=_integer("AI_MAX_TOKENS", 1000),
            ai_image_max_dimension=_integer("AI_IMAGE_MAX_DIMENSION", 1920, 320),
            ai_image_jpeg_quality=quality,
            ai_image_max_bytes=_integer("AI_IMAGE_MAX_BYTES", 5_000_000, 100_000),
            criteria_file=Path(os.getenv("CRITERIA_FILE", "criteria.txt")).expanduser(),
            state_db=Path(os.getenv("STATE_DB", "data/parking_score.db")).expanduser(),
            cache_dir=Path(os.getenv("CACHE_DIR", "data/cache")).expanduser(),
            image_extensions=extensions,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            healthcheck_max_age_seconds=_integer("HEALTHCHECK_MAX_AGE_SECONDS", 600),
        )

    def ensure_runtime_dirs(self) -> None:
        self.state_db.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
