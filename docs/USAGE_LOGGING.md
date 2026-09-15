# Central usage logging

All migrated writers use `llmpedia.papers.token_usage_logs`. `project_id` identifies the application; `process_id` identifies its task. Keep both. Task names are not globally unique.

| Writer | Project ID |
|---|---|
| Sentiment | `sentiment` |
| Dailies | `dailies` |
| Paper workflows | `llmpedia_workflows` |
| Social workflows | `llmpedia_social` |
| Direct Pi | The validated logging profile's project ID |

Old records without proven ownership retain a NULL project ID. The dashboard displays them as **Unassigned**. The retired Code/llmpedia app and the unused TypeScript logger were not migrated.

## Query and storage contracts

```python
from llm_query_utils import postgres_usage_callback, run_query, set_usage_callback

set_usage_callback(postgres_usage_callback(
    project_id="sentiment",
    connect=get_connection,
))
result = run_query(
    user_message=message,
    model=ResponseSchema,
    llm_model="gpt-5.5",
    use_pi_sdk=True,
    use_agent_sdk=False,
    pi_options={"provider": "openai-codex", "thinking": "medium", "timeout": 900},
    process_id="sentiment_analyze",
)
```

The application owns connection configuration. Registration does not open a connection. Each logged callback opens, commits, and closes one connection. Events without a process ID do not open a connection. Database errors propagate; a failed insert does not retry the LLM call.

`run_query(..., return_usage=True)` returns `QueryRun(result, usage)` on the Pi route. Other routes reject that option. It does not replace or temporarily swap the registered callback. Pi usage is emitted before structured-output validation, so a charged response can be logged even if its JSON is invalid.

The shared INSERT function accepts rows in `usage_db.USAGE_COLUMNS` order. It does not commit or close a caller-owned connection. UUID conflicts are ignored for event replay. Ordinary callbacks generate an ID per event; they do not provide a durable queue. The direct-Pi uploader preserves its existing durable pending files, IDs, destination checks, and deletion-after-commit behavior. Do not enable a second Pi logging extension on Python query calls.

## Prices are estimates, not bills

`total_cost` is the authoritative whole-call API-equivalent estimate. NULL means the total is unverified. A reported zero is different from an unknown price.

- LiteLLM and Codex input costs can already include cache charges. Their total is the calculator's input plus output, without adding cache again.
- Agent SDK reports a whole-call total.
- Pi reports `usage.cost.total`. When it is absent, a total can be derived only if all relevant component prices are known. A missing price on a known zero-token component contributes no charge; missing tokens and missing price leave the total unknown.
- Unknown Codex model pricing does not establish a free call, even if old component fields contain placeholder zeroes.

The raw component columns remain for compatibility and investigation. They do not have one universal additive meaning across historical transports. Readers must sum `total_cost`, and report `count(*) FILTER (WHERE total_cost IS NULL)` separately. Do not replace an all-NULL dollar sum with zero. Reasoning-token counts are metadata, not another charge to add to output costs.

Only proven history receives totals. The sentiment migration distinguishes positive GPT-5.5 Pi component prices from the old local GPT-5 placeholder-zero path. Records that cannot establish a complete total remain NULL. Historical direct-Pi attribution requires an explicit profile, matching process prefix, nonempty session ID, and provider-qualified model. Ambiguous generic process names stay unassigned. No operation prices old tokens using today's price catalog.

The zero-cache historical backfill verifies API-equivalent estimates, not invoices. It applies only to unassigned rows; assigned unknowns remain under their writer or migration's provenance rules. Both cache token counts must be zero, both cache costs are NULL or zero, and prompt/completion tokens and prices form complete, finite, nonnegative pairs. Positive usage requires a positive price; zero usage requires a zero price. All-zero placeholders remain unverified. With zero cache, both additive and cache-inclusive contracts reduce to prompt plus completion. The source evidence is retired `llmpedia` commit `5ea5e35`, `llmpedia_workflows` commit `417f124`, and this utility's commits `a12dd10`, `70ef403`, and `4ed474f`. It changes only a NULL `total_cost` and is safe to rerun.

## Local editable installation

The approved deployment uses `/Users/manager/Code/llm-query-utils` as the shared editable source. Updating it can affect multiple applications. Release reviewed code at a safe batch boundary and check each actual interpreter's import origin before resuming work.

Use `uv pip install --python <application>/.venv/bin/python -e '/Users/manager/Code/llm-query-utils[pi-logging]'` for a focused shared-package installation. Dailies/social manifests use `../llm-query-utils`; sentiment/engine uses `../../llm-query-utils` in its normal checkout layout.

Paper workflows has an older full requirements list with unrelated conflicts. This change updates its shared-LLM declarations; it does not make the whole legacy environment reproducible. Do not run a full requirements synchronization as part of this rollout. Compare package checks before and after, and stop on new unrelated conflicts.

Nested worktrees do not share the normal sibling layout. For tests, install the utility worktree explicitly and use `uv run --no-sync` afterwards. Confirm `llm_query_utils.__file__` points into that worktree. Generate portable lockfiles in a temporary mirror of the normal checkout layout; never commit absolute worktree paths. Social has no project build backend and runs from its repository, so `uv sync` installs dependencies without requiring an editable social package.

## Migration commands

`scripts/consolidate_usage.py` never reads `.env`. Supply an explicit DSN through `USAGE_MIGRATION_DSN` or `--dsn`. Do not print credentials or enable shell tracing. Every write requires `--apply`; copying never retires the source.

```bash
# Run in the released llm-query-utils checkout.
uv run --no-sync python scripts/consolidate_usage.py schema
uv run --no-sync python scripts/consolidate_usage.py schema --apply
uv run --no-sync python scripts/consolidate_usage.py copy
uv run --no-sync python scripts/consolidate_usage.py copy --apply
uv run --no-sync python scripts/consolidate_usage.py verify --output "$EVIDENCE/final-source.json"

# The profile itself supplies the project identity. No credential file is loaded.
uv run --no-sync python scripts/consolidate_usage.py attribute-pi \
  --profile /Users/manager/Logs/llmpedia/.pi/usage-logging.json
# Review the matched count and evidence, then repeat with --apply if approved.

# Preview the matched count and prompt-plus-completion sum. Then apply explicitly.
uv run --no-sync python scripts/consolidate_usage.py backfill-zero-cache
uv run --no-sync python scripts/consolidate_usage.py backfill-zero-cache --apply
```

The additive SQL is `migrations/001_central_usage.sql`. It expects the existing papers table. For a fresh sentiment installation on this shared database, provision that canonical contract rather than rerunning the superseded sentiment migration 017 that creates the public duplicate. Do not rewrite migration 017 in place.

Copying preserves UUIDs, UTC instants, process/model identity, token counts, raw nullable costs, and reasoning counts. Same-ID conflicts stop the copy; reruns skip identical records. Cost comparisons allow 1e-12 per field and 1e-6 across the copied source subset for float-to-numeric conversion. Differences above those bounds require investigation. NULL counts and non-cost fields must match exactly.

## Post-commit rollout

Do not run this during implementation. Review and commit all six repositories first. The source checkouts contain other edits; stop if safe delivery would overwrite them. Never point live workflows at temporary worktrees.

1. Inspect current writers and actual database destinations. Record current package/import origins and relevant wf state. Back up both usage tables to private, non-Git storage. Apply the additive schema only after confirming the backup.
2. Deliver the shared utility and consumer changes at safe batch boundaries. Do not interrupt an LLM response. Install the local editable source in each actual interpreter and compare dependency checks to its baseline. Do not fix unrelated package drift here.
3. Run copy dry-run, then initial copy. Hold sentiment writes at a safe boundary, run the final delta, and verify. Switch its released code to the canonical writer before resuming. Other projects can keep writing to papers during the copy.
4. Observe newly labeled records from each active writer. Preserve task/model settings. Do not trigger a posting or collection job just to test logging. If an application has no scheduled call yet, keep its observation gate pending.
5. Take a **fresh final backup** after public source writes have stopped. The restore proof must match the final source fingerprint, not an earlier backup made while sentiment was still writing.
6. Restore that backup into a separate disposable database. Generate restore evidence from that database, as shown below. Preview historical attribution, apply only proven rules, then deploy the homepage and verify filtered totals against SQL.
7. Retire the public table only after all evidence gates pass. The tool locks the source, rechecks the current fingerprint and row parity, requires a newly observed canonical sentiment call, and drops without CASCADE. Existing dependencies cause refusal. A source change requires fresh copy, backup, restore, and verification.

Use the actual wf utility, not an assumed Bash executable:

```bash
WF=(bash /Users/manager/Code/wf/scripts/workflows.sh)
"${WF[@]}" status --json project:sentiment
"${WF[@]}" status --json project:dailies
"${WF[@]}" status --json project:llmpedia_workflows
"${WF[@]}" status --json project:llmpedia_social
# Stop/start only the required selectors after confirming a safe boundary.
```

The original paths are sentiment/engine, dailies, llmpedia_workflows, llmpedia_social, and webdev/homepage under `/Users/manager/Code`. W runs `.venv/bin/python`; the others commonly use `uv run`. Probe the interpreter shown by current wf metadata. Also inspect the installed `pi-logged` executable's interpreter, because it may use a separate environment.

```bash
# Substitute each real application interpreter; no LLM is invoked.
uv run --no-sync python -c \
  'import llm_query_utils as q; print(q.__file__); assert callable(q.postgres_usage_callback)'
```

### Backup and restore evidence

Use protected storage, not a tracked directory. `pg_dump` and `pg_restore` must be compatible with the server version. On this machine, the complete server toolchain is `/opt/homebrew/opt/postgresql@16/bin`; PATH's libpq initdb lacks a postgres server binary.

```bash
umask 077
PG=/opt/homebrew/opt/postgresql@16/bin
# EVIDENCE and BACKUP are private paths chosen for this rollout.
# SOURCE_DSN and RESTORE_DSN are explicit, DIFFERENT databases.
"$PG/pg_dump" -Fc --dbname "$SOURCE_DSN" \
  --table=public.token_usage_logs --table=papers.token_usage_logs --file "$BACKUP"
# Create an empty disposable restore database first, with its papers schema.
"$PG/psql" "$RESTORE_DSN" -v ON_ERROR_STOP=1 -c 'CREATE SCHEMA papers'
"$PG/pg_restore" --exit-on-error --dbname "$RESTORE_DSN" "$BACKUP"
USAGE_MIGRATION_DSN="$RESTORE_DSN" uv run --no-sync python scripts/consolidate_usage.py \
  backup-evidence --backup "$BACKUP" --output "$EVIDENCE/restored-backup.json"
USAGE_MIGRATION_DSN="$SOURCE_DSN" uv run --no-sync python scripts/consolidate_usage.py \
  verify --output "$EVIDENCE/final-source.json"

# CUTOVER_AT is the actual timezone-aware release timestamp, not an inferred date.
USAGE_MIGRATION_DSN="$SOURCE_DSN" uv run --no-sync python scripts/consolidate_usage.py retire \
  --backup "$BACKUP" --verification "$EVIDENCE/final-source.json" \
  --restore-evidence "$EVIDENCE/restored-backup.json" --writers-quiescent \
  --cutover-at "$CUTOVER_AT" --apply
```

Retirement requires matching backup checksum, a separate restored-database identity, matching final source fingerprints, complete reconciliation, and a new canonical sentiment UUID outside the copied source set. `--writers-quiescent` is the operator's confirmation that the code/workflow scan found no remaining source writers; the database cannot identify arbitrary future external SQL. Do not bypass that scan because the lock succeeds.

### Read-only release checks

```sql
SELECT project_id, process_id, model_name, count(*) AS calls,
       count(total_cost) AS verified_calls,
       count(*) FILTER (WHERE total_cost IS NULL) AS unverified_calls,
       sum(total_cost) AS verified_estimate
FROM papers.token_usage_logs
WHERE tstp >= :cutover_utc
GROUP BY project_id, process_id, model_name;

SELECT to_regclass('public.token_usage_logs'); -- NULL only after retirement
```

For homepage reconciliation, use its exact date/project/noise filters. All-project verified estimates must equal the sum of project and Unassigned estimates; unverified calls must reconcile separately. The older manager dashboard still sums raw historical components and is not the oracle for verified totals.

Deploy homepage through its existing Vercel project only after the schema/API contract is available. Follow the clean-export pattern in `scripts/publish-agent-atlas.sh`, without invoking that unrelated publishing workflow:

```bash
set -euo pipefail
cd /Users/manager/Code/webdev/homepage
# HEAD must be the reviewed/released main commit containing the costs change.
test -f .vercel/project.json || exit 1
DEPLOY_DIR=$(mktemp -d /tmp/homepage-costs-deploy.XXXXXX)
git archive HEAD | tar -x -C "$DEPLOY_DIR"
cp -R .vercel "$DEPLOY_DIR/.vercel"
(cd "$DEPLOY_DIR" && vercel deploy --prod --yes)
rm -rf "$DEPLOY_DIR"
curl --fail --silent --show-error 'https://mg3.dev/api/costs?days=14&project=sentiment'
```

Check `/api/costs` and `/costs`, including identical task names in separate projects, NULL projects, unknown prices, and zero prices. Confirm private-data routes/pools are unchanged. Do not deploy the dirty live directory or a temporary development worktree.

### Rollback

Before retirement, keep both tables and all canonical rows. Restore prior application versions only through safe deployment and preserve other edits. Reconcile UUIDs again before another copy. Do not delete central rows to make totals agree.

After retirement, reverting to an old public-table writer requires restoring the public backup and reconciling any post-backup records. Preserve canonical data throughout. If recovery would lose newer records or overwrite another person's changes, stop and request a recovery decision. Keep backups until the operator approves their removal.
