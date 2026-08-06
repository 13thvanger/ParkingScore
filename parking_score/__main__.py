from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigurationError, Settings
from .service import ParkingScoreService, healthcheck, run_forever


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParkingScore FTP worker")
    parser.add_argument(
        "command", choices=("run", "once", "healthcheck"), nargs="?", default="run"
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    try:
        settings = Settings.from_env(Path(arguments.env_file))
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if arguments.command == "healthcheck":
        raise SystemExit(0 if healthcheck(settings) else 1)
    if arguments.command == "once":
        service = ParkingScoreService(settings)
        try:
            service.run_cycle()
        finally:
            service.close()
        return
    run_forever(settings)


if __name__ == "__main__":
    main()
