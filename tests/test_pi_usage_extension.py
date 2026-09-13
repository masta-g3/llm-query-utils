import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import textwrap
import uuid

import pytest


ROOT = Path(__file__).parents[1]
EXTENSION = ROOT / "src" / "llm_query_utils" / "pi_usage_extension.ts"
_DEFAULT_CONFIG = object()


def run_node(tmp_path, body, *, config=_DEFAULT_CONFIG, extra_env=None):
    if config is _DEFAULT_CONFIG:
        config = {
            "project_id": "atlas",
            "profile_path": "/project/.pi/usage-logging.json",
            "spool_dir": str(tmp_path / "pending"),
            "destination": "db.example:5432/usage/papers.token_usage_logs",
            "python": "/venv/bin/python",
        }
    source = base64.b64encode(EXTENSION.read_bytes()).decode()
    script = textwrap.dedent(
        f"""
        const extension = await import('data:text/javascript;base64,{source}');
        const handlers = new Map();
        const execCalls = [];
        const pi = {{
          on(name, handler) {{
            if (!handlers.has(name)) handlers.set(name, []);
            handlers.get(name).push(handler);
          }},
          async exec(command, args, options) {{
            execCalls.push({{command, args, options}});
            return {{stdout: '', stderr: 'database unavailable', code: 1, killed: false}};
          }},
        }};
        const ctx = {{
          model: {{provider: 'catalog-provider', id: 'catalog-model'}},
          sessionManager: {{getSessionId() {{ return 'session-123'; }}}},
        }};
        const emit = async (name, event = {{}}) => {{
          for (const handler of handlers.get(name) || []) await handler(event, ctx);
        }};
        extension.default(pi);
        {body}
        """
    )
    env = os.environ.copy()
    for name in (
        "PI_SUBAGENT_AGENT",
        "PI_TMUX_SUBAGENTS_JOB_ID",
        "PI_TMUX_SUBAGENTS_PARENT_ID",
    ):
        env.pop(name, None)
    if config is None:
        env.pop("LLM_USAGE_CONFIG", None)
    else:
        env["LLM_USAGE_CONFIG"] = json.dumps(config)
    env.update(extra_env or {})
    return subprocess.run(
        ["node", "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def pending_records(tmp_path):
    return [json.loads(path.read_text()) for path in sorted((tmp_path / "pending").glob("*.json"))]


def assistant(*, timestamp, model="requested", response_model=None, usage=None, stop_reason="stop"):
    message = {
        "role": "assistant",
        "provider": "provider",
        "model": model,
        "timestamp": timestamp,
        "stopReason": stop_reason,
        "content": [{"type": "text", "text": "SECRET RESPONSE"}],
        "usage": usage,
    }
    if response_model is not None:
        message["responseModel"] = response_model
    return message


def test_collects_each_assistant_response_and_filters_tool_cycle_events(tmp_path):
    first = assistant(
        timestamp=1_700_000_000_000,
        usage={
            "input": 10,
            "output": 0,
            "cacheRead": 4,
            "cacheWrite": 2,
            "totalTokens": 16,
            "cost": {"input": 0.01, "output": 0, "cacheRead": None},
        },
        stop_reason="toolUse",
    )
    second = assistant(
        timestamp=1_700_000_001_000,
        model="requested-2",
        response_model="served-2",
        usage={"input": 20, "output": 3, "cacheRead": 0, "cacheWrite": 0},
        stop_reason="error",
    )
    body = f"""
    const first = {json.dumps(first)};
    const second = {json.dumps(second)};
    await emit('message_end', {{message: first}});
    await emit('message_end', {{message: {{role: 'toolResult', usage: {{input: 999}}, content: ['SECRET TOOL']}}}});
    await emit('turn_end', {{message: first}});
    await emit('message_end', {{message: second}});
    await emit('message_end', {{message: first}}); // repeated callback, not a new attempt
    await emit('agent_end');
    console.log(JSON.stringify({{handlerCounts: Object.fromEntries([...handlers].map(([k,v]) => [k,v.length])), execCalls}}));
    """
    result = run_node(
        tmp_path,
        body,
        extra_env={
            "PI_SUBAGENT_AGENT": "research",
            "PI_TMUX_SUBAGENTS_JOB_ID": "job-7",
            "PI_TMUX_SUBAGENTS_PARENT_ID": "parent-3",
        },
    )

    assert result.returncode == 0, result.stderr
    records = sorted(pending_records(tmp_path), key=lambda record: record["tstp"])
    assert len(records) == 2
    assert stat.S_IMODE((tmp_path / "pending").stat().st_mode) == 0o700
    for record_path in (tmp_path / "pending").glob("*.json"):
        assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    first_record, second_record = records
    uuid.UUID(first_record["id"])
    uuid.UUID(second_record["id"])
    assert first_record["id"] != second_record["id"]
    assert first_record.keys() == {
        "id", "tstp", "model_name", "process_id", "session_id", "usage",
        "event_kind", "attribution", "requested_model", "job_id",
        "parent_job_id", "destination",
    }
    assert first_record["tstp"] == "2023-11-14T22:13:20.000Z"
    assert first_record["model_name"] == "provider/requested"
    assert first_record["process_id"] == "atlas/research"
    assert first_record["session_id"] == "session-123"
    assert first_record["event_kind"] == "response"
    assert first_record["attribution"] == "request_model"
    assert first_record["requested_model"] == "requested"
    assert first_record["job_id"] == "job-7"
    assert first_record["parent_job_id"] == "parent-3"
    assert first_record["usage"] == {
        "input": 10,
        "output": 0,
        "cacheRead": 4,
        "cacheWrite": 2,
        "totalTokens": 16,
        "cost": {"input": 0.01, "output": 0, "cacheRead": None},
    }
    assert second_record["model_name"] == "provider/served-2"
    assert second_record["attribution"] == "response_model"
    serialized = json.dumps(records)
    assert "SECRET" not in serialized
    harness = json.loads(result.stdout)
    assert "turn_end" not in harness["handlerCounts"]
    assert len(harness["execCalls"]) == 1
    assert harness["execCalls"][0] == {
        "command": "/venv/bin/python",
        "args": [
            "-m", "llm_query_utils.pi_logging", "_flush-worker", "--config",
            "/project/.pi/usage-logging.json",
        ],
        "options": {"timeout": 8000},
    }
    assert "usage upload failed" in result.stderr


def test_duplicate_factory_registration_is_ignored(tmp_path):
    message = assistant(timestamp=1_700_000_000_000, usage={"input": 1, "output": 2})
    result = run_node(
        tmp_path,
        f"""
        extension.default(pi);
        await emit('message_end', {{message: {json.dumps(message)}}});
        console.log(JSON.stringify(Object.fromEntries([...handlers].map(([k,v]) => [k,v.length]))));
        """,
    )

    assert result.returncode == 0, result.stderr
    records = pending_records(tmp_path)
    assert len(records) == 1
    assert records[0]["process_id"] == "atlas/scheduler"
    assert records[0]["job_id"] is None
    assert records[0]["parent_job_id"] is None
    assert "cost" not in records[0]["usage"]
    assert json.loads(result.stdout)["message_end"] == 1


def test_missing_usage_warns_without_saving_and_scheduler_is_default_stage(tmp_path):
    message = assistant(timestamp=1_700_000_000_000, usage=None)
    result = run_node(tmp_path, f"await emit('message_end', {{message: {json.dumps(message)}}});")

    assert result.returncode == 0
    assert pending_records(tmp_path) == []
    assert "assistant usage missing" in result.stderr


def test_default_compaction_uses_captured_catalog_model_and_flushes(tmp_path):
    usage = {"input": 100, "output": 9, "cacheRead": 5, "cacheWrite": 0, "cost": {"input": 0.2}}
    entry = {
        "type": "compaction",
        "id": "compact-1",
        "timestamp": 1_700_000_005_000,
        "usage": usage,
        "summary": "SECRET SUMMARY",
    }
    result = run_node(
        tmp_path,
        f"""
        await emit('session_before_compact', {{}});
        ctx.model = {{provider: 'changed', id: 'changed'}};
        const event = {{compactionEntry: {json.dumps(entry)}, fromExtension: false}};
        await emit('session_compact', event);
        await emit('session_compact', event);
        console.log(JSON.stringify(execCalls));
        """,
        extra_env={"PI_SUBAGENT_AGENT": "writer"},
    )

    assert result.returncode == 0, result.stderr
    records = pending_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["tstp"] == "2023-11-14T22:13:25.000Z"
    assert record["model_name"] == "catalog-provider/catalog-model"
    assert record["requested_model"] == "catalog-model"
    assert record["process_id"] == "atlas/writer/compaction"
    assert record["event_kind"] == "compaction"
    assert record["attribution"] == "compaction_catalog_model"
    assert record["usage"] == usage
    assert "SECRET" not in json.dumps(record)
    assert len(json.loads(result.stdout)) == 1


@pytest.mark.parametrize("from_extension,from_hook", [(True, False), (False, True)])
def test_custom_compaction_warns_and_does_not_invent_attribution(tmp_path, from_extension, from_hook):
    entry = {
        "type": "compaction",
        "id": "custom-1",
        "timestamp": 1_700_000_005_000,
        "usage": {"input": 8, "output": 2},
        "fromHook": from_hook,
    }
    result = run_node(
        tmp_path,
        f"""
        await emit('session_before_compact', {{}});
        await emit('session_compact', {{compactionEntry: {json.dumps(entry)}, fromExtension: {str(from_extension).lower()}}});
        console.log(JSON.stringify(execCalls));
        """,
    )

    assert result.returncode == 0
    assert pending_records(tmp_path) == []
    assert "custom compaction usage attribution unsupported" in result.stderr
    assert len(json.loads(result.stdout)) == 1


def test_compaction_without_usage_warns_and_flushes(tmp_path):
    result = run_node(
        tmp_path,
        """
        await emit('session_before_compact', {});
        await emit('session_compact', {compactionEntry: {id: 'compact-no-usage', timestamp: 1700000005000}, fromExtension: false});
        console.log(JSON.stringify(execCalls));
        """,
    )

    assert result.returncode == 0
    assert pending_records(tmp_path) == []
    assert "compaction usage missing" in result.stderr
    assert len(json.loads(result.stdout)) == 1


def test_flush_failures_never_escape_lifecycle_handlers(tmp_path):
    result = run_node(
        tmp_path,
        """
        await emit('agent_end');
        await emit('session_shutdown');
        console.log(JSON.stringify(execCalls));
        """,
    )

    assert result.returncode == 0
    assert len(json.loads(result.stdout)) == 2
    assert result.stderr.count("usage upload failed") == 2


def test_killed_upload_warns_even_when_pi_reports_exit_zero(tmp_path):
    result = run_node(
        tmp_path,
        """
        pi.exec = async () => ({code: 0, killed: true, stdout: '', stderr: ''});
        await emit('agent_end');
        """,
    )
    assert result.returncode == 0
    assert "usage upload timed out" in result.stderr


def test_missing_config_is_fatal_during_factory_initialization(tmp_path):
    result = run_node(tmp_path, "console.log('unreachable');", config=None)

    assert result.returncode == 74
    assert "invalid LLM_USAGE_CONFIG" in result.stderr
    assert "unreachable" not in result.stdout


def test_local_capture_failure_exits_immediately(tmp_path):
    spool_file = tmp_path / "not-a-directory"
    spool_file.write_text("occupied")
    config = {
        "project_id": "atlas",
        "profile_path": "/project/.pi/usage-logging.json",
        "spool_dir": str(spool_file),
        "destination": "destination",
        "python": "/venv/bin/python",
    }
    message = assistant(timestamp=1_700_000_000_000, usage={"input": 1, "output": 1})
    result = run_node(
        tmp_path,
        f"""
        await emit('message_end', {{message: {json.dumps(message)}}});
        console.log('subsequent-work-ran');
        """,
        config=config,
    )

    assert result.returncode == 74
    assert "usage capture failed" in result.stderr
    assert "subsequent-work-ran" not in result.stdout
