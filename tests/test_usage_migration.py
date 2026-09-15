import importlib.util
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest
from psycopg2.extensions import parse_dsn

SCRIPT = Path(__file__).parents[1] / "scripts" / "consolidate_usage.py"
spec = importlib.util.spec_from_file_location("consolidate_usage", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


@pytest.fixture
def conn():
    dsn = os.environ.get("USAGE_MIGRATION_TEST_DSN")
    if not dsn:
        pytest.skip("set explicit USAGE_MIGRATION_TEST_DSN to a disposable local database")
    params = parse_dsn(dsn)
    assert params.get("dbname") == "usage_test"
    assert params.get("host", "").startswith("/tmp/")
    connection = psycopg2.connect(dsn)
    with connection.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS papers CASCADE; DROP TABLE IF EXISTS public.token_usage_logs CASCADE; CREATE SCHEMA papers")
        cur.execute("""
            CREATE TABLE papers.token_usage_logs (
              id uuid PRIMARY KEY, tstp timestamp NOT NULL, model_name varchar(255) NOT NULL,
              process_id varchar(255), prompt_tokens integer, completion_tokens integer,
              prompt_cost numeric, completion_cost numeric,
              cache_creation_input_tokens integer, cache_read_input_tokens integer,
              cache_creation_cost numeric, cache_read_cost numeric, session_id varchar(255)
            );
            CREATE TABLE public.token_usage_logs (
              id uuid PRIMARY KEY, tstp timestamptz NOT NULL, model_name text NOT NULL,
              process_id text NOT NULL, prompt_tokens integer, completion_tokens integer,
              prompt_cost double precision, completion_cost double precision,
              cache_creation_input_tokens integer NOT NULL DEFAULT 0,
              cache_read_input_tokens integer NOT NULL DEFAULT 0,
              cache_creation_cost double precision, cache_read_cost double precision,
              reasoning_output_tokens integer NOT NULL DEFAULT 0
            )
        """)
    connection.commit()
    yield connection
    connection.rollback()
    connection.close()


def add_source(conn, *, process="sentiment_analyze", costs=(0.1, 0.2), model="gpt-5.5"):
    row_id = str(uuid4())
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO public.token_usage_logs
            (id,tstp,model_name,process_id,prompt_tokens,completion_tokens,prompt_cost,completion_cost)
            VALUES (%s,'2026-09-01T02:30:00+02:00',%s,%s,100,20,%s,%s)""",
            (row_id, model, process, *costs))
    return row_id


def test_schema_copy_dry_run_and_repeat(conn):
    migration.apply_schema(conn)
    migration.apply_schema(conn)
    source_id = add_source(conn)
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'America/Los_Angeles'")
    before = migration.copy_usage(conn, apply=False)
    assert before["missing"] == 1
    assert before["source_count"] == 1
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers.token_usage_logs")
        assert cur.fetchone()[0] == 0
    migration.copy_usage(conn, apply=True)
    report = migration.verify_usage(conn)
    assert report["missing"] == 0 and report["conflicts"] == 0
    assert migration.copy_usage(conn, apply=True)["inserted"] == 0
    with conn.cursor() as cur:
        cur.execute("SELECT project_id,tstp,total_cost,session_id,reasoning_output_tokens FROM papers.token_usage_logs WHERE id=%s", (source_id,))
        assert cur.fetchone() == ("sentiment", datetime(2026, 9, 1, 0, 30), Decimal("0.3"), None, 0)


def test_batched_copy_preserves_old_canonical_writer_rows(conn):
    migration.apply_schema(conn)
    legacy_id = str(uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO papers.token_usage_logs(id,tstp,model_name,process_id,prompt_cost,completion_cost) VALUES (%s,now(),'legacy-model','legacy-task',1,2)", (legacy_id,))
        from psycopg2.extras import execute_values
        execute_values(cur, """INSERT INTO public.token_usage_logs
            (id,tstp,model_name,process_id,prompt_tokens,completion_tokens,prompt_cost,completion_cost)
            VALUES %s""", [(str(uuid4()), datetime(2026, 9, 1, tzinfo=timezone.utc), "gpt-5.5", "sentiment_analyze", 100, 20, 0.1, 0.2) for _ in range(1003)])
    report = migration.copy_usage(conn, apply=True)
    assert report["inserted"] == report["source_count"] == 1003
    assert report["costs_match"] and report["nulls_match"]
    with conn.cursor() as cur:
        cur.execute("SELECT project_id,total_cost,prompt_cost,completion_cost FROM papers.token_usage_logs WHERE id=%s", (legacy_id,))
        assert cur.fetchone() == (None, None, Decimal(1), Decimal(2))
    assert migration.copy_usage(conn, apply=True)["inserted"] == 0


def test_unknown_prices_and_placeholder_zeros_stay_unknown(conn):
    migration.apply_schema(conn)
    ids = [add_source(conn, costs=(0, 0)), add_source(conn, costs=(0.1, None)), add_source(conn, model="unknown-model")]
    migration.copy_usage(conn, apply=True)
    with conn.cursor() as cur:
        cur.execute("SELECT total_cost FROM papers.token_usage_logs WHERE id=ANY(%s::uuid[])", (ids,))
        assert cur.fetchall() == [(None,), (None,), (None,)]
    assert migration.verify_usage(conn)["unverified_totals"] == 3


def test_collision_and_unexpected_source_refused(conn):
    migration.apply_schema(conn)
    row_id = add_source(conn)
    migration.copy_usage(conn, apply=True)
    with conn.cursor() as cur:
        cur.execute("UPDATE papers.token_usage_logs SET prompt_tokens=999 WHERE id=%s", (row_id,))
    assert migration.verify_usage(conn)["conflicts"] == 1
    with pytest.raises(ValueError, match="conflict"):
        migration.copy_usage(conn, apply=True)
    add_source(conn, process="unrelated_process")
    with pytest.raises(ValueError, match="Unexpected source"):
        migration.verify_usage(conn)


def test_empty_late_delta_and_null_parity(conn):
    migration.apply_schema(conn)
    assert migration.verify_usage(conn)["source_count"] == 0
    first = add_source(conn, costs=(None, None))
    migration.copy_usage(conn, apply=True)
    add_source(conn)
    assert migration.verify_usage(conn)["missing"] == 1
    migration.copy_usage(conn, apply=True)
    assert migration.verify_usage(conn)["missing"] == 0
    with conn.cursor() as cur:
        cur.execute("SELECT prompt_cost,completion_cost,total_cost FROM papers.token_usage_logs WHERE id=%s", (first,))
        assert cur.fetchone() == (None, None, None)


def test_zero_cache_backfill_accepts_only_complete_historical_estimates(conn):
    migration.apply_schema(conn)
    rows = [
        # label, prompt tokens/cost, completion tokens/cost, creation tokens/cost,
        # read tokens/cost, existing total
        ("valid", 100, "0.25", 20, "0.50", 0, None, 0, "0", None),
        ("valid-zero-component", 0, "0", 20, "0.50", 0, "0", 0, None, None),
        ("partial-price", 100, None, 20, "0.50", 0, None, 0, None, None),
        ("nullable-token-count", None, "0.25", 20, "0.50", 0, None, 0, None, None),
        ("negative-token-count", -1, "0.25", 20, "0.50", 0, None, 0, None, None),
        ("nullable-cache-count", 100, "0.25", 20, "0.50", None, None, 0, None, None),
        ("positive-cache-tokens", 100, "0.25", 20, "0.50", 1, None, 0, None, None),
        ("nonzero-cache-cost", 100, "0.25", 20, "0.50", 0, "0.01", 0, None, None),
        ("zero-price-positive-usage", 100, "0", 20, "0.50", 0, None, 0, None, None),
        ("zero-token-positive-cost", 0, "0.25", 20, "0.50", 0, None, 0, None, None),
        ("all-zero", 0, "0", 0, "0", 0, None, 0, None, None),
        ("nan", 100, "NaN", 20, "0.50", 0, None, 0, None, None),
        ("infinity", 100, "Infinity", 20, "0.50", 0, None, 0, None, None),
        ("negative", 100, "-0.25", 20, "0.50", 0, None, 0, None, None),
        ("known-total", 100, "0.25", 20, "0.50", 0, None, 0, None, "9.99"),
        ("assigned-unknown", 100, "0.25", 20, "0.50", 0, None, 0, None, None),
    ]
    with conn.cursor() as cur:
        for row in rows:
            label, *values = row
            cur.execute("""INSERT INTO papers.token_usage_logs
                (id,tstp,model_name,process_id,project_id,prompt_tokens,prompt_cost,
                 completion_tokens,completion_cost,cache_creation_input_tokens,
                 cache_creation_cost,cache_read_input_tokens,cache_read_cost,total_cost)
                VALUES (%s,now(),%s,'legacy',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (str(uuid4()), label, "keep-me" if label in {"known-total", "assigned-unknown"} else None, *values))

    with conn.cursor() as cur:
        cur.execute("""SELECT model_name,project_id,prompt_tokens,prompt_cost::text,completion_tokens,
            completion_cost::text,cache_creation_input_tokens,cache_creation_cost::text,
            cache_read_input_tokens,cache_read_cost::text FROM papers.token_usage_logs ORDER BY model_name""")
        raw_before = cur.fetchall()
    before = migration.backfill_zero_cache(conn, apply=False)
    assert before == {"matched": 2, "total_cost_sum": Decimal("1.25"), "updated": 0, "dry_run": True}
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers.token_usage_logs WHERE total_cost IS NOT NULL")
        assert cur.fetchone()[0] == 1

    applied = migration.backfill_zero_cache(conn, apply=True)
    assert applied == {"matched": 2, "total_cost_sum": Decimal("1.25"), "updated": 2, "dry_run": False}
    with conn.cursor() as cur:
        cur.execute("SELECT model_name,project_id,total_cost,prompt_tokens,prompt_cost FROM papers.token_usage_logs ORDER BY model_name")
        stored = {row[0]: row[1:] for row in cur.fetchall()}
        cur.execute("""SELECT model_name,project_id,prompt_tokens,prompt_cost::text,completion_tokens,
            completion_cost::text,cache_creation_input_tokens,cache_creation_cost::text,
            cache_read_input_tokens,cache_read_cost::text FROM papers.token_usage_logs ORDER BY model_name""")
        assert cur.fetchall() == raw_before
    assert stored["valid"] == (None, Decimal("0.75"), 100, Decimal("0.25"))
    assert stored["valid-zero-component"] == (None, Decimal("0.50"), 0, Decimal("0"))
    assert stored["assigned-unknown"][0:2] == ("keep-me", None)
    assert stored["known-total"][1] == Decimal("9.99")
    assert all(stored[label][1] is None for label, *_ in rows if label not in {"valid", "valid-zero-component", "known-total"})
    assert migration.backfill_zero_cache(conn, apply=True) == {
        "matched": 0, "total_cost_sum": Decimal(0), "updated": 0, "dry_run": False,
    }


def test_profile_attribution_requires_session_and_preserves_other_projects(conn, tmp_path):
    migration.apply_schema(conn)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"project_id": "llmpedia", "table": "papers.token_usage_logs"}))
    with conn.cursor() as cur:
        for session, project in [("session-a", None), (None, None), ("session-b", "different")]:
            cur.execute("""INSERT INTO papers.token_usage_logs
              (id,tstp,model_name,process_id,session_id,project_id,prompt_tokens,completion_tokens,
               cache_creation_input_tokens,cache_read_input_tokens,prompt_cost,completion_cost)
              VALUES (%s,now(),'openai-codex/gpt-5.5','llmpedia/scheduler',%s,%s,10,2,0,0,0.1,0.2)""",
              (str(uuid4()),session,project))
    with pytest.raises(ValueError, match="project"):
        migration.attribute_pi(conn, profile, apply=True)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM papers.token_usage_logs WHERE project_id='different'")
    assert migration.attribute_pi(conn, profile, apply=False)["matched"] == 1
    assert migration.attribute_pi(conn, profile, apply=True)["updated"] == 1
    with conn.cursor() as cur:
        cur.execute("SELECT project_id,total_cost FROM papers.token_usage_logs ORDER BY session_id NULLS LAST")
        assert cur.fetchall() == [("llmpedia", Decimal("0.3")), (None, None)]


def test_profile_attribution_batches_and_preserves_totals(conn, tmp_path):
    from psycopg2.extras import execute_values

    migration.apply_schema(conn)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"project_id": "llmpedia", "table": "papers.token_usage_logs"}))
    # Sorted IDs put an all-NULL total batch first and the conflict last.
    rows = [(f"{i:032x}", None, 10, None, None) for i in range(1, 1001)]
    rows += [
        (f"{1001:032x}", None, 0, None, None),
        (f"{1002:032x}", None, 10, Decimal("0.1"), Decimal("9.5")),
        (f"{1003:032x}", None, 10, Decimal("0.1"), None),
        (f"{1004:032x}", "different", 10, None, None),
    ]
    with conn.cursor() as cur:
        execute_values(cur, """INSERT INTO papers.token_usage_logs
            (id,project_id,prompt_tokens,prompt_cost,total_cost,tstp,model_name,process_id,
             session_id,completion_tokens,cache_creation_input_tokens,cache_read_input_tokens)
            VALUES %s""", rows,
            template="(%s,%s,%s,%s,%s,now(),'openai-codex/gpt-5.5','llmpedia/scheduler','session',0,0,0)")
        cur.execute("""
            CREATE TEMP TABLE attribution_updates (statement_count integer);
            CREATE FUNCTION pg_temp.record_attribution_update() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                INSERT INTO attribution_updates VALUES (1);
                RETURN NULL;
            END $$;
            CREATE TRIGGER count_attribution_updates AFTER UPDATE ON papers.token_usage_logs
                FOR EACH STATEMENT EXECUTE FUNCTION pg_temp.record_attribution_update()
        """)
    for apply in (False, True):
        with pytest.raises(ValueError, match="Conflicting historical project"):
            migration.attribute_pi(conn, profile, apply=apply)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM attribution_updates")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM papers.token_usage_logs WHERE project_id IS NULL")
        assert cur.fetchone()[0] == 1003
        cur.execute("DELETE FROM papers.token_usage_logs WHERE project_id='different'")
        cur.execute("SELECT * FROM papers.token_usage_logs ORDER BY id")
        before = cur.fetchall()
    report = migration.attribute_pi(conn, profile, apply=False)
    assert (report["matched"], report["would_update"], report["updated"], report["dry_run"]) == (1003, 1003, 0, True)
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM papers.token_usage_logs ORDER BY id")
        assert cur.fetchall() == before
        cur.execute("SELECT count(*) FROM attribution_updates")
        assert cur.fetchone()[0] == 0
    report = migration.attribute_pi(conn, profile, apply=True)
    assert (report["matched"], report["would_update"], report["updated"], report["dry_run"]) == (1003, 1003, 1003, False)
    with conn.cursor() as cur:
        cur.execute("SELECT project_id,total_cost FROM papers.token_usage_logs ORDER BY id")
        assert cur.fetchall() == [("llmpedia", None)] * 1000 + [
            ("llmpedia", Decimal(0)), ("llmpedia", Decimal("9.5")), ("llmpedia", Decimal("0.1")),
        ]
        cur.execute("SELECT count(*) FROM attribution_updates")
        statement_count = cur.fetchone()[0]
        assert 1 <= statement_count <= 2
    report = migration.attribute_pi(conn, profile, apply=True)
    assert (report["matched"], report["would_update"], report["updated"]) == (1003, 0, 0)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM attribution_updates")
        assert cur.fetchone()[0] == statement_count


def test_retirement_refuses_missing_proof_and_live_delta(conn, tmp_path):
    migration.apply_schema(conn)
    add_source(conn)
    migration.copy_usage(conn, apply=True)
    report = migration.verify_usage(conn)
    with pytest.raises(ValueError, match="backup"):
        migration.retire_source(conn, report, {}, tmp_path / "missing.dump", writers_quiescent=True, cutover_at=datetime.now(timezone.utc))
    add_source(conn)
    with pytest.raises(ValueError):
        migration.retire_source(conn, report, {}, tmp_path / "missing.dump", writers_quiescent=True, cutover_at=datetime.now(timezone.utc))


@pytest.fixture
def restored_backup(conn, tmp_path):
    migration.apply_schema(conn)
    add_source(conn)
    migration.copy_usage(conn, apply=True)
    conn.commit()
    dsn = os.environ["USAGE_MIGRATION_TEST_DSN"]
    pg = Path("/opt/homebrew/opt/postgresql@16/bin")
    if not (pg / "pg_dump").is_file():
        pytest.skip("local PostgreSQL 16 tools required for restore rehearsal")
    backup = tmp_path / "usage.dump"
    subprocess.run([str(pg / "pg_dump"), "-Fc", "--dbname", dsn,
                    "--table=public.token_usage_logs", "--table=papers.token_usage_logs", "--file", str(backup)], check=True, capture_output=True)
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS usage_restore_test")
        cur.execute("CREATE DATABASE usage_restore_test")
    params = parse_dsn(dsn) | {"dbname": "usage_restore_test"}
    restored = psycopg2.connect(**params)
    with restored.cursor() as cur:
        cur.execute("CREATE SCHEMA papers")
    restored.commit()
    subprocess.run([str(pg / "pg_restore"), "--exit-on-error", "--dbname", restored.dsn, str(backup)], check=True, capture_output=True)
    evidence = migration.backup_evidence(restored, backup)
    restored.close()
    yield backup, evidence
    with admin.cursor() as cur:
        cur.execute("DROP DATABASE usage_restore_test")
    admin.close()


def add_canonical_call(conn, since):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO papers.token_usage_logs
            (id,tstp,model_name,process_id,project_id,total_cost)
            VALUES (%s,%s,'gpt-5.5','sentiment_analyze','sentiment',0.1)""", (str(uuid4()), since.astimezone(timezone.utc).replace(tzinfo=None)))


def test_real_backup_restore_and_retirement_guards(conn, restored_backup):
    backup, evidence = restored_backup
    report = migration.verify_usage(conn)
    since = datetime.now(timezone.utc) - timedelta(minutes=1)
    with pytest.raises(ValueError, match="quiescent"):
        migration.retire_source(conn, report, evidence, backup, writers_quiescent=False, cutover_at=since)
    with pytest.raises(ValueError, match="No new canonical"):
        migration.retire_source(conn, report, evidence, backup, writers_quiescent=True, cutover_at=since)
    add_canonical_call(conn, datetime.now(timezone.utc))
    result = migration.retire_source(conn, report, evidence, backup, writers_quiescent=True, cutover_at=since)
    assert result["retired"]
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.token_usage_logs'),count(*) FROM papers.token_usage_logs")
        assert cur.fetchone() == (None, 2)
    conn.commit()


def test_retirement_rejects_stale_backup_and_changed_source(conn, restored_backup):
    backup, evidence = restored_backup
    old_report = migration.verify_usage(conn)
    add_source(conn)
    migration.copy_usage(conn, apply=True)
    since = datetime.now(timezone.utc) - timedelta(minutes=1)
    add_canonical_call(conn, datetime.now(timezone.utc))
    with pytest.raises(ValueError, match="Source changed"):
        migration.retire_source(conn, old_report, evidence, backup, writers_quiescent=True, cutover_at=since)
    new_report = migration.verify_usage(conn)
    with pytest.raises(ValueError, match="Source changed"):
        migration.retire_source(conn, new_report, evidence, backup, writers_quiescent=True, cutover_at=since)
    conn.rollback()


def test_retirement_never_cascades_dependencies(conn, restored_backup):
    backup, evidence = restored_backup
    add_canonical_call(conn, datetime.now(timezone.utc))
    with conn.cursor() as cur:
        cur.execute("CREATE VIEW public.usage_dependency AS SELECT * FROM public.token_usage_logs")
    conn.commit()
    report = migration.verify_usage(conn)
    with pytest.raises(psycopg2.errors.DependentObjectsStillExist):
        migration.retire_source(conn, report, evidence, backup, writers_quiescent=True,
                                cutover_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.usage_dependency")
        assert cur.fetchone()[0] == 1
