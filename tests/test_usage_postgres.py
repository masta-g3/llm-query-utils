"""Explicit disposable-PostgreSQL checks; never load a project's .env."""
import json
import os
from datetime import datetime, timezone
from decimal import Decimal

import psycopg2
import pytest
from psycopg2.extensions import parse_dsn

from llm_query_utils import UsageData, postgres_usage_callback
from llm_query_utils import pi_logging
from test_pi_logging import make_record


@pytest.fixture
def database():
    dsn = os.environ.get("USAGE_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set USAGE_POSTGRES_TEST_DSN to a disposable local usage_shared database")
    params = parse_dsn(dsn)
    assert params["dbname"] == "usage_shared" and params["host"].startswith("/tmp/")
    conn = psycopg2.connect(dsn)
    with conn, conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS papers CASCADE; CREATE SCHEMA papers")
        cur.execute("""CREATE TABLE papers.token_usage_logs (
            id uuid PRIMARY KEY, tstp timestamp NOT NULL, model_name varchar(255) NOT NULL,
            process_id varchar(255), session_id varchar(255), prompt_tokens integer,
            completion_tokens integer, prompt_cost numeric, completion_cost numeric,
            cache_creation_input_tokens integer, cache_read_input_tokens integer,
            cache_creation_cost numeric, cache_read_cost numeric, project_id text,
            reasoning_output_tokens integer, total_cost numeric)""")
    yield params, conn
    conn.close()


def test_real_callback_projects_nulls_and_utc(database):
    params, read = database
    opened = []

    def connect():
        conn = psycopg2.connect(**params, options="-c timezone=America/Los_Angeles")
        opened.append(conn)
        return conn

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    for project, total in [("sentiment", Decimal("0.3")), ("dailies", None)]:
        callback = postgres_usage_callback(project, connect)
        callback(UsageData("gpt-5.5", None, 100, 20, 0.1, 0.2))
        callback(UsageData("gpt-5.5", "same_task", 100, 20, 0.1, 0.2,
                           cache_read_input_tokens=5, cache_read_cost=0.01,
                           reasoning_output_tokens=3, total_cost=total))
    assert len(opened) == 2 and all(conn.closed for conn in opened)
    with read.cursor() as cur:
        cur.execute("SELECT project_id,process_id,total_cost,reasoning_output_tokens,tstp FROM papers.token_usage_logs ORDER BY project_id")
        rows = cur.fetchall()
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [
        ("dailies", "same_task", None, 3), ("sentiment", "same_task", Decimal("0.3"), 3)]
    assert all(before <= row[4] <= datetime.now(timezone.utc).replace(tzinfo=None) for row in rows)


def test_real_pi_commit_failure_retains_uuid_and_retry_is_idempotent(database, tmp_path):
    params, conn = database
    profile = pi_logging.Profile(
        project_id="example-project", path=tmp_path / "profile.json", env_file=tmp_path / ".env",
        table="papers.token_usage_logs", database=params, spool_dir=tmp_path,
        destination=pi_logging.destination_fingerprint(params["host"], params["port"], params["dbname"], "papers.token_usage_logs"),
    )
    record = make_record(profile)
    record["usage"] = {"reasoning": 4, "cost": {"total": 0.75}}
    pending = tmp_path / f"{record['id']}.json"
    pending.write_text(json.dumps(record))
    with conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE papers.projects (id text PRIMARY KEY)")
        cur.execute("ALTER TABLE papers.token_usage_logs ADD CONSTRAINT project_fk FOREIGN KEY(project_id) REFERENCES papers.projects(id) DEFERRABLE INITIALLY DEFERRED")
    failed = pi_logging.flush_pending(profile)
    assert failed.errors == 1 and pending.exists()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers.token_usage_logs")
        assert cur.fetchone()[0] == 0
        cur.execute("INSERT INTO papers.projects VALUES ('example-project')")
    assert pi_logging.flush_pending(profile).sent == 1
    assert not pending.exists()
    pending.write_text(json.dumps(record))
    assert pi_logging.flush_pending(profile).sent == 1
    assert not pending.exists()
    with conn.cursor() as cur:
        cur.execute("SELECT id::text,project_id,total_cost,reasoning_output_tokens,prompt_tokens FROM papers.token_usage_logs")
        assert cur.fetchall() == [(record["id"], "example-project", Decimal("0.75"), 4, None)]
