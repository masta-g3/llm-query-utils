"""Consolidate sentiment usage without guessing historical prices.

Every command requires an explicit DSN. Writes require --apply. This script
never loads .env and never drops the source as a side effect of copying it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "migrations/001_central_usage.sql"
PROCESSES = ("sentiment_analyze", "sentiment_context_labels", "sentiment_batch_context_labels")
COMPONENTS = (
    ("prompt_cost", "prompt_tokens"),
    ("completion_cost", "completion_tokens"),
    ("cache_creation_cost", "cache_creation_input_tokens"),
    ("cache_read_cost", "cache_read_input_tokens"),
)
COST_FIELDS = tuple(cost for cost, _ in COMPONENTS)
FIELDS = (
    "id", "tstp", "model_name", "process_id", "prompt_tokens", "completion_tokens",
    "prompt_cost", "completion_cost", "cache_creation_input_tokens", "cache_read_input_tokens",
    "cache_creation_cost", "cache_read_cost", "reasoning_output_tokens", "session_id",
    "project_id", "total_cost",
)
FIELD_TOLERANCE = Decimal("1e-12")
TOTAL_TOLERANCE = Decimal("1e-6")
SOURCE_SELECT = """SELECT id::text, tstp AT TIME ZONE 'UTC' AS tstp,
    model_name, process_id, prompt_tokens, completion_tokens,
    prompt_cost::numeric, completion_cost::numeric,
    cache_creation_input_tokens, cache_read_input_tokens,
    cache_creation_cost::numeric, cache_read_cost::numeric,
    reasoning_output_tokens
    FROM public.token_usage_logs ORDER BY id"""
TARGET_SELECT = ", ".join("id::text" if name == "id" else name for name in FIELDS)
ZERO_CACHE_BACKFILL_PREDICATE = """project_id IS NULL AND total_cost IS NULL
    AND cache_creation_input_tokens=0 AND cache_read_input_tokens=0
    AND (cache_creation_cost IS NULL OR cache_creation_cost=0)
    AND (cache_read_cost IS NULL OR cache_read_cost=0)
    AND prompt_tokens IS NOT NULL AND prompt_tokens>=0
    AND completion_tokens IS NOT NULL AND completion_tokens>=0
    AND prompt_cost IS NOT NULL AND prompt_cost>=0 AND prompt_cost<'Infinity'::numeric
    AND completion_cost IS NOT NULL AND completion_cost>=0 AND completion_cost<'Infinity'::numeric
    AND ((prompt_tokens>0 AND prompt_cost>0) OR (prompt_tokens=0 AND prompt_cost=0))
    AND ((completion_tokens>0 AND completion_cost>0) OR (completion_tokens=0 AND completion_cost=0))
    AND prompt_cost+completion_cost>0"""


def apply_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text())


def database_identity(conn):
    params = conn.get_dsn_parameters()
    return {key: params.get(key) for key in ("host", "port", "dbname")}


def source_fingerprint(conn):
    digest = hashlib.sha256()
    with conn.cursor() as cur:
        cur.execute("SET LOCAL TIME ZONE 'UTC'")
    with conn.cursor(name="usage_fingerprint") as cur:
        cur.itersize = 1000
        cur.execute("SELECT row_to_json(row)::text FROM (SELECT * FROM public.token_usage_logs ORDER BY id) row")
        for (row,) in cur:
            digest.update(row.encode())
            digest.update(b"\n")
    return digest.hexdigest()


def complete_component_total(row):
    total = Decimal(0)
    for cost, tokens in COMPONENTS:
        value = row[cost]
        if value is None:
            if row[tokens] != 0:
                return None
        else:
            if not value.is_finite() or value < 0:
                raise ValueError("Non-finite or negative historical cost requires investigation")
            total += value
    return total


def sentiment_row(row):
    if row["process_id"] not in PROCESSES:
        raise ValueError(f"Unexpected source process: {row['process_id']}")
    row = dict(row)
    row.update(session_id=None, project_id="sentiment", total_cost=None)
    # The old local GPT-5 calculator emitted placeholder zero prices. Positive
    # GPT-5.5 prices came from the later Pi mapping; other provenance is unknown.
    if row["model_name"] == "gpt-5.5" and any((row[key] or 0) > 0 for key in COST_FIELDS):
        row["total_cost"] = complete_component_total(row)
    return row


def source_batches(conn):
    with conn.cursor(name="usage_source", cursor_factory=RealDictCursor) as cur:
        cur.execute(SOURCE_SELECT)
        while batch := cur.fetchmany(1000):
            yield [sentiment_row(row) for row in batch]


def target_rows(conn, ids):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT {TARGET_SELECT} FROM papers.token_usage_logs WHERE id=ANY(%s::uuid[])", (ids,))
        return {row["id"]: row for row in cur}


def equivalent(source, target):
    for key in FIELDS:
        a, b = source[key], target[key]
        if key in (*COST_FIELDS, "total_cost") and a is not None and b is not None:
            if not a.is_finite() or not b.is_finite() or abs(a - b) > FIELD_TOLERANCE:
                return False
        elif a != b:
            return False
    return True


def verify_usage(conn):
    report = {
        "database": database_identity(conn), "source_count": 0,
        "missing": 0, "conflicts": 0, "verified_totals": 0, "unverified_totals": 0,
    }
    source_sums = {key: Decimal(0) for key in COST_FIELDS}
    target_sums = {key: Decimal(0) for key in COST_FIELDS}
    source_nulls = {key: 0 for key in COST_FIELDS}
    target_nulls = {key: 0 for key in COST_FIELDS}
    for batch in source_batches(conn):
        targets = target_rows(conn, [row["id"] for row in batch])
        for source in batch:
            report["source_count"] += 1
            report["verified_totals" if source["total_cost"] is not None else "unverified_totals"] += 1
            target = targets.get(source["id"])
            if target is None:
                report["missing"] += 1
            elif not equivalent(source, target):
                report["conflicts"] += 1
            for key in COST_FIELDS:
                source_sums[key] += source[key] or 0
                source_nulls[key] += source[key] is None
                if target is not None:
                    target_sums[key] += target[key] or 0
                    target_nulls[key] += target[key] is None
    report.update(source_fingerprint=source_fingerprint(conn), source_cost_sums=source_sums,
                  target_cost_sums=target_sums, source_nulls=source_nulls, target_nulls=target_nulls)
    report["costs_match"] = all(abs(source_sums[key] - target_sums[key]) <= TOTAL_TOLERANCE for key in COST_FIELDS)
    report["nulls_match"] = source_nulls == target_nulls
    return report


def require_verified(report):
    if report["missing"] or report["conflicts"] or not report["costs_match"] or not report["nulls_match"]:
        raise ValueError("Source reconciliation failed: missing rows, conflicts, costs, or NULLs differ")


def copy_usage(conn, *, apply=False):
    before = verify_usage(conn)
    if before["conflicts"]:
        raise ValueError("Existing ID conflict; no copy is allowed")
    if not apply:
        return before | {"inserted": 0, "dry_run": True}
    inserted = 0
    for batch in source_batches(conn):
        existing = target_rows(conn, [row["id"] for row in batch])
        new = [row for row in batch if row["id"] not in existing]
        if new:
            with conn.cursor() as cur:
                execute_values(cur, f"INSERT INTO papers.token_usage_logs ({', '.join(FIELDS)}) VALUES %s ON CONFLICT (id) DO NOTHING",
                               [tuple(row[key] for key in FIELDS) for row in new], page_size=1000)
                inserted += cur.rowcount
    after = verify_usage(conn)
    require_verified(after)
    return after | {"inserted": inserted, "dry_run": False}


def attribute_pi(conn, profile_path, *, apply=False):
    profile = json.loads(Path(profile_path).read_text())
    project = profile.get("project_id")
    if not isinstance(project, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project):
        raise ValueError("Invalid profile project_id")
    if profile.get("table") != "papers.token_usage_logs":
        raise ValueError("Profile must target papers.token_usage_logs")
    # A process prefix alone does not prove ownership. The old direct-Pi
    # uploader also wrote session_id and a provider-qualified model name.
    evidence = """split_part(process_id,'/',1)=%s AND process_id LIKE '%%/%%'
        AND session_id IS NOT NULL AND session_id<>''
        AND model_name LIKE '%%/%%'"""
    with conn.cursor() as cur:
        cur.execute(f"""SELECT EXISTS (SELECT 1 FROM papers.token_usage_logs
            WHERE {evidence} AND project_id IS NOT NULL AND project_id<>%s)""", (project, project))
        if cur.fetchone()[0]:
            raise ValueError("Conflicting historical project attribution")
    matched = would_update = 0
    with conn.cursor(name="usage_pi_attribution", cursor_factory=RealDictCursor) as source:
        source.execute(f"SELECT {TARGET_SELECT} FROM papers.token_usage_logs WHERE {evidence} ORDER BY id", (project,))
        while batch := source.fetchmany(1000):
            matched += len(batch)
            updates = []
            for row in batch:
                total = row["total_cost"]
                if total is None:
                    total = complete_component_total(row)
                if row["project_id"] is None or row["total_cost"] != total:
                    updates.append((project, total, row["id"]))
            would_update += len(updates)
            if apply and updates:
                with conn.cursor() as cur:
                    execute_values(cur, """UPDATE papers.token_usage_logs AS target
                        SET project_id=changes.project_id,total_cost=changes.total_cost
                        FROM (VALUES %s) AS changes(project_id,total_cost,id)
                        WHERE target.id=changes.id""", updates,
                        template="(%s,%s::numeric,%s::uuid)", page_size=1000)
    return {"project_id": project, "matched": matched, "would_update": would_update,
            "updated": would_update if apply else 0, "dry_run": not apply,
            "evidence": "explicit profile + matching project prefix + nonempty Pi session + provider/model"}


def backfill_zero_cache(conn, *, apply=False):
    if apply:
        sql = f"""WITH changed AS (
            UPDATE papers.token_usage_logs
            SET total_cost=prompt_cost+completion_cost
            WHERE {ZERO_CACHE_BACKFILL_PREDICATE}
            RETURNING total_cost)
            SELECT count(*),coalesce(sum(total_cost),0) FROM changed"""
    else:
        sql = f"""SELECT count(*),coalesce(sum(prompt_cost+completion_cost),0)
            FROM papers.token_usage_logs WHERE {ZERO_CACHE_BACKFILL_PREDICATE}"""
    with conn.cursor() as cur:
        cur.execute(sql)
        matched, total = cur.fetchone()
    return {"matched": matched, "total_cost_sum": total,
            "updated": matched if apply else 0, "dry_run": not apply}


def backup_checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_evidence(conn, backup):
    backup = Path(backup)
    if not backup.is_file() or not backup.stat().st_size:
        raise ValueError("A nonempty backup file is required")
    # Run this ONLY against the separate database restored from this backup.
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM papers.token_usage_logs LIMIT 1")
    return {"backup_sha256": backup_checksum(backup), "restored_database": database_identity(conn),
            "source_fingerprint": source_fingerprint(conn)}


def retire_source(conn, verification, restore_evidence, backup, *, writers_quiescent, cutover_at):
    backup = Path(backup)
    if not backup.is_file() or not backup.stat().st_size:
        raise ValueError("A restored backup is required before retirement")
    if not writers_quiescent:
        raise ValueError("Confirm source writers are quiescent before retirement")
    backup_hash = backup_checksum(backup)
    if restore_evidence.get("backup_sha256") != backup_hash:
        raise ValueError("Backup checksum does not match restore evidence")
    identity = database_identity(conn)
    if restore_evidence.get("restored_database") in (None, identity):
        raise ValueError("Backup must be verified in a separate restored database")
    if verification.get("database") != identity:
        raise ValueError("Verification belongs to a different source database")
    if cutover_at is None or cutover_at.tzinfo is None:
        raise ValueError("An explicit timezone-aware cutover time is required")
    with conn.cursor() as cur:
        cur.execute("LOCK TABLE public.token_usage_logs IN ACCESS EXCLUSIVE MODE")
    current = verify_usage(conn)
    require_verified(current)
    fingerprint = current["source_fingerprint"]
    if verification.get("source_fingerprint") != fingerprint or restore_evidence.get("source_fingerprint") != fingerprint:
        raise ValueError("Source changed since reconciliation or restored backup; repeat both")
    with conn.cursor() as cur:
        cur.execute("""SELECT count(*) FROM papers.token_usage_logs p
            WHERE p.project_id='sentiment' AND p.process_id=ANY(%s)
              AND p.tstp >= %s AND NOT EXISTS(SELECT 1 FROM public.token_usage_logs s WHERE s.id=p.id)""",
            (list(PROCESSES), cutover_at.astimezone(timezone.utc).replace(tzinfo=None)))
        if not cur.fetchone()[0]:
            raise ValueError("No new canonical sentiment call observed after cutover")
        # PostgreSQL refuses dependent views/FKs; never use CASCADE.
        cur.execute("DROP TABLE public.token_usage_logs")
    return {"retired": True, "source_count": current["source_count"], "backup_sha256": backup_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("schema", "copy", "verify", "attribute-pi", "backfill-zero-cache", "backup-evidence", "retire"))
    parser.add_argument("--dsn", default=os.environ.get("USAGE_MIGRATION_DSN"), help="explicit DSN or USAGE_MIGRATION_DSN; .env is not loaded")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--restore-evidence", type=Path)
    parser.add_argument("--writers-quiescent", action="store_true")
    parser.add_argument("--cutover-at", type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("an explicit --dsn or USAGE_MIGRATION_DSN is required")
    if args.action == "attribute-pi" and args.profile is None:
        parser.error("attribute-pi requires --profile")
    if args.action == "backup-evidence" and args.backup is None:
        parser.error("backup-evidence requires --backup and the restored database DSN")
    if args.action == "retire" and args.apply and not all((args.backup, args.verification, args.restore_evidence, args.cutover_at)):
        parser.error("retire --apply requires --backup, --verification, --restore-evidence, and --cutover-at")
    conn = psycopg2.connect(args.dsn, connect_timeout=10)
    conn.set_session(isolation_level="REPEATABLE READ")
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='120s'; SET LOCAL TIME ZONE 'UTC'")
            if args.action == "schema":
                if args.apply:
                    apply_schema(conn)
                result = {"schema_applied": args.apply, "sql": SCHEMA_SQL.read_text()}
            elif args.action == "copy":
                result = copy_usage(conn, apply=args.apply)
            elif args.action == "verify":
                result = verify_usage(conn)
                require_verified(result)
            elif args.action == "attribute-pi":
                result = attribute_pi(conn, args.profile, apply=args.apply)
            elif args.action == "backfill-zero-cache":
                result = backfill_zero_cache(conn, apply=args.apply)
            elif args.action == "backup-evidence":
                result = backup_evidence(conn, args.backup)
            elif not args.apply:
                result = verify_usage(conn) | {"retired": False, "dry_run": True}
            else:
                result = retire_source(conn, json.loads(args.verification.read_text()), json.loads(args.restore_evidence.read_text()),
                                       args.backup, writers_quiescent=args.writers_quiescent, cutover_at=args.cutover_at)
        text = json.dumps(result, default=str, indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        print(text, end="")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
