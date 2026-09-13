"""Transparent Pi launcher and durable PostgreSQL usage delivery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .usage import pi_total_cost
from .usage_db import insert_usage_rows

PROFILE_NAME = Path(".pi/usage-logging.json")
ALLOWED_TABLE = "papers.token_usage_logs"
BATCH_SIZE = 100
CONNECT_TIMEOUT = 3
STATEMENT_TIMEOUT_MS = 3000
UPLOAD_TIMEOUT = 8
MAX_DB_INTEGER = 2_147_483_647


class ConfigError(ValueError):
    """The project logging profile is absent or unsafe."""


class RecordError(ValueError):
    """A pending record does not satisfy the spool contract."""


@dataclass(frozen=True)
class Profile:
    project_id: str
    path: Path
    env_file: Path
    table: str
    database: dict[str, str]
    spool_dir: Path
    destination: str


@dataclass(frozen=True)
class FlushResult:
    sent: int = 0
    invalid: int = 0
    errors: int = 0


def destination_fingerprint(host: str, port: str, dbname: str, table: str) -> str:
    payload = json.dumps(
        {"host": host, "port": str(port), "dbname": dbname, "table": table},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def load_profile(path: str | os.PathLike[str]) -> Profile:
    profile_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(profile_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read usage logging profile: {profile_path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("usage logging profile must be a JSON object")

    project_id = raw.get("project_id")
    if not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project_id):
        raise ConfigError("invalid project_id in usage logging profile")
    if raw.get("table") != ALLOWED_TABLE:
        raise ConfigError(f"usage logging table must be {ALLOWED_TABLE}")
    env_setting = raw.get("env_file")
    if not isinstance(env_setting, str) or not env_setting:
        raise ConfigError("usage logging profile requires env_file")
    env_file = (profile_path.parent / env_setting).resolve()

    try:
        import psycopg2  # noqa: F401
        from dotenv import dotenv_values
    except ImportError as exc:
        raise ConfigError("Pi logging dependencies are not installed; install the pi-logging extra") from exc
    # Credential files are authoritative. Keep substitutions literal instead of
    # resolving them from the launcher's ambient environment.
    values = dotenv_values(env_file, interpolate=False)
    fields = {
        "dbname": values.get("DB_NAME"),
        "user": values.get("DB_USER"),
        "password": values.get("DB_PASS"),
        "host": values.get("DB_HOST"),
        "port": values.get("DB_PORT"),
    }
    missing = [name for name, value in fields.items() if value is None or value == ""]
    if missing:
        raise ConfigError(f"usage logging env file is missing required database fields: {', '.join(missing)}")
    database = {name: str(value) for name, value in fields.items()}

    spool_dir = Path.home() / ".local/state/llm-query-utils/pi" / project_id
    _private_directory(spool_dir)
    destination = destination_fingerprint(
        database["host"], database["port"], database["dbname"], ALLOWED_TABLE
    )
    return Profile(
        project_id=project_id,
        path=profile_path,
        env_file=env_file,
        table=ALLOWED_TABLE,
        database=database,
        spool_dir=spool_dir,
        destination=destination,
    )


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RecordError("tstp must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordError("tstp must be an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RecordError("tstp must be an ISO UTC timestamp")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _number(mapping: dict[str, Any], key: str, *, integer: bool = False) -> int | float | None:
    value = mapping.get(key)
    if value is None:
        return None
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise RecordError(f"usage.{key} must be numeric")
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise RecordError(f"usage.{key} must be finite nonnegative numeric data")
    if integer and value > MAX_DB_INTEGER:
        raise RecordError(f"usage.{key} exceeds the database integer range")
    return value


def record_values(
    record: dict[str, Any], destination: str, project_id: str | None = None
) -> tuple[Any, ...]:
    if not isinstance(record, dict):
        raise RecordError("record must be a JSON object")
    if record.get("destination") != destination:
        raise RecordError("pending record destination does not match profile")
    try:
        record_id = uuid.UUID(record["id"])
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise RecordError("record id must be a UUID") from exc
    record_id_text = str(record_id)
    if record_id_text != record.get("id"):
        raise RecordError("record id must use canonical UUID form")

    for key in ("model_name", "process_id", "session_id", "requested_model"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise RecordError(f"{key} must be a non-empty string")
    for key in ("model_name", "process_id", "session_id"):
        if len(record[key]) > 255:
            raise RecordError(f"{key} must not exceed 255 characters")
    if "/" not in record["model_name"]:
        raise RecordError("model_name must contain provider/model")
    if record.get("event_kind") not in {"response", "compaction"}:
        raise RecordError("event_kind is invalid")
    if record.get("attribution") not in {
        "response_model", "request_model", "compaction_catalog_model"
    }:
        raise RecordError("attribution is invalid")
    for key in ("job_id", "parent_job_id"):
        if record.get(key) is not None and not isinstance(record.get(key), str):
            raise RecordError(f"{key} must be a string or null")
    usage = record.get("usage")
    if not isinstance(usage, dict):
        raise RecordError("usage must be an object")
    cost = usage.get("cost")
    if cost is None:
        cost = {}
    if not isinstance(cost, dict):
        raise RecordError("usage.cost must be an object")

    _number(cost, "total")  # Reject corrupt pending data before deriving a total.
    total_cost = pi_total_cost(usage)

    return (
        record_id_text,
        _utc_timestamp(record.get("tstp")),
        record["model_name"],
        record["process_id"],
        record.get("session_id"),
        _number(usage, "input", integer=True),
        _number(usage, "output", integer=True),
        _number(cost, "input"),
        _number(cost, "output"),
        _number(usage, "cacheWrite", integer=True),
        _number(usage, "cacheRead", integer=True),
        _number(cost, "cacheWrite"),
        _number(cost, "cacheRead"),
        project_id,
        _number(usage, "reasoning", integer=True),
        total_cost,
    )


def read_record(
    path: Path,
    destination: str,
    project_id: str | None = None,
    *,
    raw: Any = None,
) -> tuple[Any, ...]:
    try:
        filename_id = uuid.UUID(path.stem)
    except ValueError as exc:
        raise RecordError("pending record filename must be a UUID") from exc
    if str(filename_id) != path.stem:
        raise RecordError("pending record filename must use canonical UUID form")
    if raw is None:
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise RecordError("pending record is not valid JSON") from exc
    values = record_values(raw, destination, project_id)
    if values[0] != str(filename_id):
        raise RecordError("pending record filename does not match record id")
    return values


def _default_connect(**kwargs: Any) -> Any:
    try:
        import psycopg2
    except ImportError as exc:
        raise ConfigError("Pi logging dependencies are not installed; install the pi-logging extra") from exc
    return psycopg2.connect(**kwargs)


def flush_pending(
    profile: Profile,
    *,
    connect: Callable[..., Any] | None = None,
    batch_size: int = BATCH_SIZE,
) -> FlushResult:
    try:
        import psycopg2
    except ImportError as exc:
        raise ConfigError("Pi logging dependencies are not installed; install the pi-logging extra") from exc

    pending: list[tuple[Path, tuple[Any, ...]]] = []
    invalid = 0
    for path in profile.spool_dir.glob("*.json"):
        try:
            values = read_record(path, profile.destination, profile.project_id)
        except FileNotFoundError:
            # Another uploader committed and removed the immutable record.
            continue
        except RecordError as exc:
            invalid += 1
            print(f"llm-usage: invalid pending record {path.name}: {exc}", file=sys.stderr)
            continue
        pending.append((path, values))
        if len(pending) >= batch_size:
            break
    if not pending:
        return FlushResult(invalid=invalid)

    connector = connect or _default_connect
    connection = None
    try:
        connection = connector(
            **profile.database,
            connect_timeout=CONNECT_TIMEOUT,
            options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
        )
        insert_usage_rows(connection, (row for _, row in pending))
        connection.commit()
    except (psycopg2.Error, OSError):
        if connection is not None:
            try:
                connection.rollback()
            except (psycopg2.Error, OSError):
                pass
        print("llm-usage: database delivery failed; pending records retained", file=sys.stderr)
        return FlushResult(invalid=invalid, errors=1)
    finally:
        if connection is not None:
            try:
                connection.close()
            except (psycopg2.Error, OSError):
                pass

    for path, _ in pending:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            print(f"llm-usage: committed record could not be removed: {path.name}", file=sys.stderr)
            return FlushResult(sent=len(pending), invalid=invalid, errors=1)
    return FlushResult(sent=len(pending), invalid=invalid)


def extension_path() -> Path:
    resource = importlib.resources.files("llm_query_utils").joinpath("pi_usage_extension.ts")
    path = Path(str(resource)).resolve()
    if not path.is_file():
        raise ConfigError("packaged Pi usage extension is missing")
    return path


def _resolve_self(path: str | os.PathLike[str]) -> Path:
    text = os.fspath(path)
    resolved = shutil.which(text) if os.sep not in text else text
    if not resolved:
        raise ConfigError("cannot resolve pi-logged executable")
    return Path(resolved).resolve()


def launch(
    arguments: list[str],
    profile_path: str | os.PathLike[str],
    *,
    self_path: str | os.PathLike[str],
) -> None:
    profile = load_profile(profile_path)
    self_executable = _resolve_self(self_path)
    pi_value = shutil.which("pi")
    if not pi_value:
        raise ConfigError("normal pi executable was not found on PATH")
    pi_executable = Path(pi_value).resolve()
    try:
        recursive = os.path.samefile(pi_executable, self_executable)
    except OSError:
        recursive = pi_executable == self_executable
    if recursive:
        raise ConfigError("pi-logged refuses to launch itself as pi")
    extension = extension_path()

    if _bounded_flush(str(profile.path)):
        print("pi-logged: pending usage delivery failed; continuing", file=sys.stderr)
    environment = os.environ.copy()
    environment["PI_TMUX_SUBAGENTS_PI_BIN"] = str(self_executable)
    environment["LLM_USAGE_CONFIG"] = json.dumps(
        {
            "project_id": profile.project_id,
            "profile_path": str(profile.path),
            "spool_dir": str(profile.spool_dir),
            "destination": profile.destination,
            "python": os.path.abspath(sys.executable),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    argv = [str(pi_executable), "--extension", str(extension), *arguments]
    os.execvpe(str(pi_executable), argv, environment)


def main() -> int:
    try:
        profile_path = Path.cwd() / PROFILE_NAME
        launch(sys.argv[1:], profile_path, self_path=sys.argv[0])
    except ConfigError as exc:
        print(f"pi-logged: {exc}", file=sys.stderr)
        return 2
    return 0


def _flush_worker(config: str) -> int:
    try:
        profile = load_profile(config)
        result = flush_pending(profile)
    except ConfigError as exc:
        print(f"llm-usage: {exc}", file=sys.stderr)
        return 2
    return 1 if result.errors or result.invalid else 0


def _bounded_flush(config: str) -> int:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "llm_query_utils.pi_logging", "_flush-worker", "--config", config],
            timeout=UPLOAD_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("llm-usage: pending usage delivery timed out", file=sys.stderr)
        return 1
    except OSError:
        print("llm-usage: pending usage delivery could not start", file=sys.stderr)
        return 1
    return completed.returncode


def usage_main() -> int:
    parser = argparse.ArgumentParser(prog="llm-usage")
    subparsers = parser.add_subparsers(dest="command", required=True)
    flush_parser = subparsers.add_parser("flush", help="upload pending Pi usage records")
    worker_parser = subparsers.add_parser("_flush-worker", help=argparse.SUPPRESS)
    for command_parser in (flush_parser, worker_parser):
        command_parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "flush":
        return _bounded_flush(args.config)
    return _flush_worker(args.config)


if __name__ == "__main__":
    raise SystemExit(usage_main())
