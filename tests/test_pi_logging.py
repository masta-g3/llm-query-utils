import json
import os
import stat
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from llm_query_utils import pi_logging


@pytest.fixture
def profile(tmp_path, monkeypatch):
    pytest.importorskip("psycopg2", reason="run with --extra pi-logging")
    pytest.importorskip("dotenv", reason="run with --extra pi-logging")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "project"
    config_dir = root / ".pi"
    config_dir.mkdir(parents=True)
    (root / ".env").write_text(
        "DB_NAME=usage\nDB_USER=writer\nDB_PASS=secret\nDB_HOST=db.internal\nDB_PORT=5433\n"
        "IGNORED_FROM_AMBIENT=${AMBIENT_SECRET}\n"
    )
    path = config_dir / "usage-logging.json"
    path.write_text(json.dumps({
        "project_id": "example-project",
        "env_file": "../.env",
        "table": "papers.token_usage_logs",
    }))
    return path


def make_record(profile, **updates):
    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "tstp": "2026-09-12T15:04:05.123Z",
        "model_name": "openai-codex/gpt-5.4",
        "process_id": "example-project/scheduler",
        "session_id": "session-1",
        "usage": {
            "input": 11,
            "output": 7,
            "cacheRead": 5,
            "cacheWrite": 3,
            "reasoning": 2,
            "cost": {"input": 0, "output": 0.2, "cacheRead": 0.01, "total": 0.25},
        },
        "event_kind": "response",
        "attribution": "response_model",
        "requested_model": "gpt-5.4",
        "job_id": None,
        "parent_job_id": None,
        "destination": profile.destination,
    }
    record.update(updates)
    return record


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, values=None):
        self.connection.calls.append((sql, values))


class FakeConnection:
    def __init__(self, fail_commit=False):
        self.calls = []
        self.fail_commit = fail_commit
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("password=do-not-log host details")
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_profile_uses_explicit_env_and_stable_secret_free_destination(profile, monkeypatch):
    monkeypatch.setenv("DB_HOST", "wrong.ambient.host")
    loaded = pi_logging.load_profile(profile)

    assert loaded.database == {
        "dbname": "usage", "user": "writer", "password": "secret",
        "host": "db.internal", "port": "5433",
    }
    assert loaded.destination == pi_logging.destination_fingerprint(
        "db.internal", "5433", "usage", "papers.token_usage_logs"
    )
    assert "secret" not in loaded.destination
    assert stat.S_IMODE(loaded.spool_dir.stat().st_mode) == 0o700


def test_profile_does_not_interpolate_ambient_environment(profile, monkeypatch):
    monkeypatch.setenv("AMBIENT_SECRET", "ambient-value")
    env_file = profile.parent.parent / ".env"
    env_file.write_text(env_file.read_text().replace("DB_PASS=secret", "DB_PASS=${AMBIENT_SECRET}"))

    loaded = pi_logging.load_profile(profile)

    assert loaded.database["password"] == "${AMBIENT_SECRET}"


def test_profile_rejects_wrong_table_and_missing_credentials(profile):
    data = json.loads(profile.read_text())
    data["table"] = "public.token_usage_logs"
    profile.write_text(json.dumps(data))
    with pytest.raises(pi_logging.ConfigError, match="table"):
        pi_logging.load_profile(profile)


def test_record_values_map_cache_and_nullable_costs(profile):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)

    values = pi_logging.record_values(record, loaded.destination, loaded.project_id)

    assert values[0] == record["id"]
    assert isinstance(values[0], str)
    assert values[1] == datetime(2026, 9, 12, 15, 4, 5, 123000)
    assert values[5:] == (11, 7, 0, 0.2, 3, 5, None, 0.01, "example-project", 2, 0.25)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_record_rejects_invalid_numeric_values(profile, value):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)
    record["usage"]["cost"]["output"] = value

    with pytest.raises(pi_logging.RecordError, match="finite nonnegative"):
        pi_logging.record_values(record, loaded.destination)


def test_record_rejects_token_count_outside_database_range(profile):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)
    record["usage"]["input"] = 2**31

    with pytest.raises(pi_logging.RecordError, match="database integer range"):
        pi_logging.record_values(record, loaded.destination)


def test_record_rejects_database_column_overflow(profile):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded, process_id="x" * 256)

    with pytest.raises(pi_logging.RecordError, match="process_id.*255"):
        pi_logging.record_values(record, loaded.destination)


def test_record_derives_total_only_when_all_component_costs_are_complete(profile):
    loaded = pi_logging.load_profile(profile)
    complete = make_record(loaded)
    complete["usage"]["cost"].pop("total")
    complete["usage"]["cost"]["cacheWrite"] = 0.02
    incomplete = make_record(loaded)
    incomplete["usage"]["cost"].pop("total")

    assert pi_logging.record_values(complete, loaded.destination, loaded.project_id)[-1] == pytest.approx(0.23)
    assert pi_logging.record_values(incomplete, loaded.destination, loaded.project_id)[-1] is None


def test_record_rejects_filename_or_destination_mismatch(profile):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)
    with pytest.raises(pi_logging.RecordError, match="filename"):
        pi_logging.read_record(Path(f"{uuid.uuid4()}.json"), loaded.destination, raw=record)
    record["destination"] = "other"
    with pytest.raises(pi_logging.RecordError, match="destination"):
        pi_logging.record_values(record, loaded.destination)


def test_flush_commits_one_batch_then_deletes(profile):
    loaded = pi_logging.load_profile(profile)
    first = make_record(loaded)
    second = make_record(loaded)
    for record in (first, second):
        (loaded.spool_dir / f"{record['id']}.json").write_text(json.dumps(record))
    connection = FakeConnection()
    connect_calls = []

    def connect(**kwargs):
        connect_calls.append(kwargs)
        return connection

    result = pi_logging.flush_pending(loaded, connect=connect)

    assert result.sent == 2 and result.errors == 0
    assert connection.committed and connection.closed
    assert list(loaded.spool_dir.glob("*.json")) == []
    assert connect_calls[0]["connect_timeout"] == 3
    assert "statement_timeout=3000" in connect_calls[0]["options"]
    inserts = [call for call in connection.calls if call[1] is not None]
    assert len(inserts) == 1
    sql, params = inserts[0]
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert sql.count("%s") == len(params) == 32
    actual_rows = {params[i]: params[i:i + 16] for i in range(0, len(params), 16)}
    assert actual_rows == {
        record['id']: pi_logging.record_values(record, loaded.destination, loaded.project_id)
        for record in (first, second)
    }


def test_flush_failure_retains_records_and_hides_database_details(profile, capsys):
    import psycopg2

    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)
    pending = loaded.spool_dir / f"{record['id']}.json"
    pending.write_text(json.dumps(record))

    def fail_connect(**kwargs):
        raise psycopg2.OperationalError("password=do-not-log host details")

    result = pi_logging.flush_pending(loaded, connect=fail_connect)

    assert result.errors == 1 and pending.exists()
    diagnostic = capsys.readouterr().err
    assert "delivery failed" in diagnostic
    assert "do-not-log" not in diagnostic and "secret" not in diagnostic


def test_flush_does_not_swallow_programming_errors(profile):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)
    (loaded.spool_dir / f"{record['id']}.json").write_text(json.dumps(record))

    def broken_connect(**kwargs):
        raise RuntimeError("caller bug")

    with pytest.raises(RuntimeError, match="caller bug"):
        pi_logging.flush_pending(loaded, connect=broken_connect)


def test_concurrently_removed_record_is_benign(profile, monkeypatch):
    loaded = pi_logging.load_profile(profile)
    record = make_record(loaded)
    pending = loaded.spool_dir / f"{record['id']}.json"
    pending.write_text(json.dumps(record))
    original = Path.read_text

    def remove_before_read(path, *args, **kwargs):
        if path == pending:
            path.unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", remove_before_read)

    assert pi_logging.flush_pending(loaded, connect=lambda **kw: FakeConnection()) == pi_logging.FlushResult()


def test_invalid_record_is_retained_without_starving_valid_record(profile, capsys):
    loaded = pi_logging.load_profile(profile)
    (loaded.spool_dir / "not-a-uuid.json").write_text("not json")
    record = make_record(loaded)
    valid = loaded.spool_dir / f"{record['id']}.json"
    valid.write_text(json.dumps(record))
    connection = FakeConnection()

    result = pi_logging.flush_pending(loaded, connect=lambda **kw: connection)

    assert result.sent == 1 and result.invalid == 1
    assert not valid.exists()
    assert (loaded.spool_dir / "not-a-uuid.json").exists()
    assert "invalid pending record" in capsys.readouterr().err


def test_malformed_backlog_does_not_starve_valid_delivery(profile, capsys):
    loaded = pi_logging.load_profile(profile)
    for index in range(150):
        (loaded.spool_dir / f"bad-{index:04}.json").write_text("not json")
    record = make_record(loaded)
    valid = loaded.spool_dir / f"{record['id']}.json"
    valid.write_text(json.dumps(record))

    result = pi_logging.flush_pending(loaded, connect=lambda **kw: FakeConnection())

    assert result.sent == 1 and result.invalid == 150
    assert not valid.exists()
    assert len(list(loaded.spool_dir.glob('bad-*.json'))) == 150
    assert "invalid pending record" in capsys.readouterr().err


def test_flush_limits_each_database_attempt_to_100_records(profile):
    loaded = pi_logging.load_profile(profile)
    for _ in range(101):
        record = make_record(loaded)
        (loaded.spool_dir / f"{record['id']}.json").write_text(json.dumps(record))
    connection = FakeConnection()

    result = pi_logging.flush_pending(loaded, connect=lambda **kw: connection)

    assert result.sent == 100
    assert len(list(loaded.spool_dir.glob("*.json"))) == 1


@pytest.mark.parametrize("flush_status", [0, 1])
def test_launch_injects_extension_and_preserves_arguments(profile, monkeypatch, flush_status):
    loaded = pi_logging.load_profile(profile)
    fake_pi = profile.parent.parent / "bin" / "pi"
    fake_pi.parent.mkdir()
    fake_pi.write_text("#!/bin/sh\n")
    fake_pi.chmod(0o755)
    self_path = profile.parent.parent / "bin" / "pi-logged"
    self_path.write_text("#!/bin/sh\n")
    self_path.chmod(0o755)
    captured = {}
    flush_calls = []
    monkeypatch.setattr(pi_logging.shutil, "which", lambda name: str(fake_pi))
    monkeypatch.setattr(pi_logging, "extension_path", lambda: Path("/pkg/pi_usage_extension.ts"))
    monkeypatch.setattr(pi_logging.subprocess, "run", lambda *a, **kw: flush_calls.append((a, kw)) or subprocess.CompletedProcess(a[0], flush_status))
    monkeypatch.setattr(pi_logging.os, "execvpe", lambda exe, argv, env: captured.update(exe=exe, argv=argv, env=env))

    pi_logging.launch(["--print", "--no-extensions", "--", "prompt"], profile, self_path=self_path)

    assert captured["argv"] == [str(fake_pi.resolve()), "--extension", "/pkg/pi_usage_extension.ts", "--print", "--no-extensions", "--", "prompt"]
    assert flush_calls[0][1]["timeout"] == 8
    assert flush_calls[0][0][0][-2:] == ["--config", str(profile.resolve())]
    assert captured["env"]["PI_TMUX_SUBAGENTS_PI_BIN"] == str(self_path.resolve())
    config = json.loads(captured["env"]["LLM_USAGE_CONFIG"])
    assert config == {
        "project_id": "example-project",
        "profile_path": str(profile.resolve()),
        "spool_dir": str(loaded.spool_dir),
        "destination": loaded.destination,
        "python": os.path.abspath(os.sys.executable),
    }
    assert "secret" not in json.dumps(config)


def test_manual_flush_has_total_deadline(profile, monkeypatch):
    calls = []
    monkeypatch.setattr(pi_logging.subprocess, "run", lambda *a, **kw: calls.append((a, kw)) or subprocess.CompletedProcess(a[0], 0))
    monkeypatch.setattr(os.sys, "argv", ["llm-usage", "flush", "--config", str(profile)])

    assert pi_logging.usage_main() == 0
    command = calls[0][0][0]
    assert "_flush-worker" in command
    assert calls[0][1]["timeout"] == 8


def test_manual_flush_reports_total_deadline(profile, monkeypatch, capsys):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(pi_logging.subprocess, "run", time_out)
    monkeypatch.setattr(os.sys, "argv", ["llm-usage", "flush", "--config", str(profile)])

    assert pi_logging.usage_main() == 1
    assert "timed out" in capsys.readouterr().err


def test_launch_rejects_recursive_pi(profile, monkeypatch):
    self_path = profile.parent.parent / "pi"
    self_path.write_text("#!/bin/sh\n")
    self_path.chmod(0o755)
    monkeypatch.setattr(pi_logging.shutil, "which", lambda name: str(self_path))
    with pytest.raises(pi_logging.ConfigError, match="itself"):
        pi_logging.launch([], profile, self_path=self_path)
