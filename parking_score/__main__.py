from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigurationError, Settings
from .criteria import CriteriaError, load_criteria
from .database import Repository
from .evaluation import EvaluationError, EvaluationRunner
from .exporter import export_assessments_ndjson
from .service import ParkingScoreService, healthcheck, run_forever


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParkingScore FTP worker")
    parser.add_argument(
        "command",
        choices=(
            "run",
            "once",
            "healthcheck",
            "export",
            "evaluate",
            "validate-criteria",
        ),
        nargs="?",
        default="run",
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--state-db", default="data/parking_score.db")
    parser.add_argument("--after-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--criteria", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.command == "export":
        if arguments.output is None:
            print("export requires --output", file=sys.stderr)
            raise SystemExit(2)
        repository = Repository(Path(arguments.state_db))
        try:
            count, cursor = export_assessments_ndjson(
                repository,
                arguments.output,
                after_id=arguments.after_id,
                limit=arguments.limit,
            )
        except ValueError as exc:
            print(f"Export error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        finally:
            repository.close()
        print(f"exported={count} next_after_id={cursor}")
        return

    if arguments.command == "validate-criteria":
        if arguments.criteria is None:
            print("validate-criteria requires --criteria", file=sys.stderr)
            raise SystemExit(2)
        try:
            criteria = load_criteria(arguments.criteria)
        except CriteriaError as exc:
            print(f"Criteria error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        print(
            f"valid criteria_hash={criteria.content_hash} "
            f"version={criteria.version}"
        )
        return

    if arguments.command == "evaluate" and not arguments.no_publish:
        print("evaluate requires explicit --no-publish", file=sys.stderr)
        raise SystemExit(2)
    if arguments.command == "evaluate" and (
        arguments.criteria is None
        or arguments.dataset is None
        or arguments.output is None
    ):
        print(
            "evaluate requires --criteria, --dataset and --output",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        settings = Settings.from_env(Path(arguments.env_file))
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if arguments.command == "evaluate":
        try:
            criteria = load_criteria(arguments.criteria)
        except CriteriaError as exc:
            print(f"Criteria error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        runner = EvaluationRunner(settings)
        try:
            summary = runner.run(criteria, arguments.dataset, arguments.output)
        except EvaluationError as exc:
            print(f"Evaluation error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        finally:
            runner.close()
        print(
            f"evaluation_run_id={summary.evaluation_run_id} "
            f"total={summary.total} completed={summary.completed} "
            f"failed={summary.failed} skipped={summary.skipped}"
        )
        if summary.failed:
            raise SystemExit(1)
        return
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
