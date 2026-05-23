**Feature:** pi-cli-integration → Add a minimal Pi command-line transport to `run_query()` alongside Codex CLI, Claude Agent SDK, and LiteLLM.

## Summary

Implemented an explicit `use_pi_sdk=True` route backed by the local `pi` CLI in JSON mode. The route is single-turn only, defaults to ephemeral/tool-free/deterministic Pi invocation, parses final assistant text and usage from Pi JSONL events, maps Pi cost/token usage into `UsageData`, and validates Pydantic structured output from JSON text.

## Implemented Files

- `src/llm_query_utils/query.py`
  - Added Pi option validation and command construction.
  - Added Pi JSONL parsing with `message_end` preferred over `agent_end` fallback.
  - Added Pi usage mapping for input/output/cache tokens and costs.
  - Added `_run_pi_cli()` subprocess wrapper and `_pi_sdk_query()` async wrapper.
  - Added `use_pi_sdk` and `pi_options` to `run_query()` routing.
- `tests/test_query_helpers_and_utils.py`
  - Added Pi parsing, command-building, validation, usage mapping, structured output, and subprocess failure tests.
  - Updated Codex subprocess fakes for the existing timeout parameter.
- `tests/test_query_routing.py`
  - Added explicit Pi routing and rejection tests for incompatible `messages`, `sdk_allowed_tools`, and existing extended-thinking args.
- `tests/test_pi_live_regression.py`
  - Added opt-in live tests gated by `RUN_LIVE_PI_TESTS=1`.
- `README.md`, `docs/STRUCTURE.md`, `pyproject.toml`
  - Updated user-facing and architecture docs/metadata for CLI/SDK transports and Pi route behavior.

## Behavior

Default Pi command shape:

```bash
pi --mode json --no-session --no-extensions --no-skills --no-prompt-templates --no-context-files --no-tools --model <model> [--system-prompt <system>] <prompt>
```

Supported `pi_options` keys:

- `model`
- `provider`
- `thinking`
- `tools`
- `timeout`
- `cwd`
- `isolate_resources`
- `no_session`

Rejected for v1:

- multi-turn `messages`
- `sdk_allowed_tools` with Pi route
- existing Anthropic/LiteLLM extended-thinking parameters; callers should use `pi_options["thinking"]`
- unknown `pi_options` keys

## Verification

Ran targeted and full test suites successfully:

```bash
uv run pytest tests/test_query_helpers_and_utils.py -k 'pi' tests/test_query_routing.py -k 'pi'
uv run pytest
```

Final full-suite result:

```text
42 passed, 6 skipped
```

Live Pi tests were added but remain opt-in:

```bash
RUN_LIVE_PI_TESTS=1 uv run pytest tests/test_pi_live_regression.py
```

## Review Notes

Review identified and fixed high-impact issues:

- Pi option validation now rejects falsy non-dicts instead of silently accepting them.
- Pi `message_end` output remains authoritative; `agent_end.messages` is fallback only.
- Non-zero Pi error output is bounded when no structured error is parsed.

## Non-goals Preserved

- No TypeScript SDK bridge or Node dependency added.
- No temporary Pi structured-output extension added.
- No Pi tools enabled by default.
- No broad backend/routing redesign.
