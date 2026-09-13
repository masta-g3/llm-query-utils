# llm-query-utils — Architecture

## Purpose

Shared LLM query and usage-logging tools extracted from [llmpedia-workflows](https://github.com/manolorueda/llmpedia-workflows). `run_query()` routes Python queries to the requested backend. The separate opt-in `pi-logged` command records direct Pi agent usage without changing query routing.

**Target users:** Internal projects that need LLM calls with structured output, retries, caching, and usage tracking without duplicating plumbing.

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Query routing | `run_query()` | Single entrypoint, auto-selects backend |
| Pi backend | Pi CLI (`pi --mode json`) | Local Pi auth/config path from Python without a Node bridge |
| Codex backend | Codex CLI (`codex exec`) | ChatGPT subscription auth path without API keys |
| Claude backend | `claude-agent-sdk` | Native structured output, tool use |
| Other models | `litellm>=1.81.11` + `instructor` | OpenAI-compatible and `chatgpt/...` subscription access, structured output via tool-calling |
| Schemas | `pydantic` v2 | Shared structured-output validation across backends |
| Cost tracking | `litellm` pricing utilities | Per-call token/cost calculation; Codex costs are API-equivalent estimates |
| Packaging | `uv` + `uv_build` | Fast, lockfile-based |

## Architecture

```mermaid
flowchart TD
    A[run_query] --> P{use_pi_sdk?}
    P -->|Yes| Q[_pi_sdk_query]
    P -->|No| B{gpt-5* model + use_codex_sdk?}
    B -->|Yes| C[_codex_sdk_query]
    B -->|No| D{Claude model + use_agent_sdk?}
    D -->|Yes| E[_agent_sdk_query]
    D -->|No| F[LiteLLM / Instructor]

    Q --> G[Usage callback]
    C --> G
    E --> G
    F --> G
    G --> H[Consumer logs/DB]
```

### Routing rules

`run_query()` can route to four backends:
- Pi CLI when:
  - `use_pi_sdk=True`
  - single-turn `user_message` flow (no `messages`)
  - existing extended thinking options are not enabled; use `pi_options={"thinking": "high"}` instead
  - `sdk_allowed_tools` is not provided; use `pi_options={"tools": [...]}` instead
  - provider/model disambiguation is explicit via `pi_options`, for example `pi_options={"provider": "openai-codex", "timeout": 900}` for `llm_model="gpt-5.5"` when Pi should use its Codex-backed provider
- Codex CLI when:
  - `use_codex_sdk=True`
  - `llm_model` basename starts with `gpt-5`, including provider-prefixed names like `chatgpt/gpt-5.4`
  - single-turn `user_message` flow (no `messages`)
  - extended thinking options are not enabled
  - optional `codex_options={"timeout": 900}` bounds hung `codex exec` calls
- Agent SDK for Claude models (default), but falls back to LiteLLM when:
- Custom `messages` list is provided (multi-turn)
- Extended thinking is enabled
- Model name doesn't contain "claude"
- LiteLLM/Instructor for all remaining cases, including ChatGPT subscription models named `chatgpt/...` when `use_codex_sdk=False`

## Directory Layout

```
llm-query-utils/
├── pyproject.toml          # Package metadata, dependencies
├── features.yaml           # Backlog tracker
├── docs/
│   └── STRUCTURE.md        # This file
└── src/llm_query_utils/
    ├── __init__.py          # Public API exports
    ├── config.py            # DEFAULT_MODEL, FAST_MODEL constants
    ├── query.py             # run_query() + _pi_sdk_query() + _codex_sdk_query() + _agent_sdk_query()
    ├── cache.py             # add_cache_control() for Anthropic prompt caching
    ├── vision.py            # format_vision_messages() for multi-modal
    ├── pi_logging.py        # Opt-in Pi launcher and PostgreSQL uploader
    ├── pi_usage_extension.ts # Packaged response/compaction capture
    └── usage.py             # UsageData dataclass + callback mechanism
```

## Key Modules

### `query.py` — Core routing

- **`run_query()`**: Synchronous entrypoint. Routes to Pi CLI when `use_pi_sdk=True`, Codex CLI for `gpt-5*` models when `use_codex_sdk=True`, Agent SDK for Claude, or LiteLLM otherwise. Handles retries (0s, 30s, 60s, 120s), extended thinking setup, and temperature disabling for reasoning models (o1/o3/gpt-5).
- **`_pi_sdk_query()`**: Async Pi CLI transport wrapper using `pi --mode json`. Defaults to `--no-session`, `--no-tools`, and isolated resources. Supports plain output and prompt-instructed structured output validated by Pydantic, and maps Pi message usage into the usage callback. Use `pi_options.provider`, `pi_options.thinking`, and `pi_options.timeout` for provider choice, thinking level, and process bounds. `pi_options.extensions` loads explicit extension sources while keeping discovery disabled; `pi_options.tools` allowlists the tools exposed to the model.
- **`_codex_sdk_query()`**: Async Codex transport wrapper using `codex exec` + ChatGPT-auth preflight (`codex login status`). Supports plain and structured output (`--output-schema`), maps token usage from JSON events, and prices usage with LiteLLM API-equivalent estimates using the actual executed GPT-5 model. Supports `codex_options.timeout` for hard process timeouts.
- **`_agent_sdk_query()`**: Async Agent SDK handler. Streams messages, counts tool calls, extracts structured output via `output_format`.

### `usage.py` — Usage tracking

Consumer registers a callback via `set_usage_callback(fn)`. Query backends emit `UsageData` (tokens, costs, cache stats) through this callback, decoupling this package from any specific logging/DB implementation.

Usage emission notes:
- LiteLLM path emits usage from provider-reported token usage and LiteLLM pricing.
- Pi path emits usage from Pi assistant message usage payloads.
- Codex path emits exact token usage from Codex event stream payloads and fills cost fields with LiteLLM API-equivalent estimates. Those costs are reporting proxies for subscription-backed Codex calls, not actual ChatGPT subscription spend.
- If LiteLLM does not know a Codex model's pricing yet, Codex usage still emits token counts with zero costs and logs a warning.
- Agent SDK path emits one usage event when a terminal `ResultMessage` is received.
- If Agent SDK omits usage payload fields, Agent usage falls back to zeros instead of skipping the callback event.

### Direct Pi usage logging

`pi_logging.py` owns the `pi-logged` launcher, profile parsing, and `llm-usage flush` command. `pi_usage_extension.ts` ships inside the Python package and subscribes to Pi response and default-compaction events. The launcher explicitly injects this extension and sets the tmux child executable to itself; it does not modify global Pi settings.

```text
pi-logged -> normal Pi + packaged extension
                         -> private pending JSON files
                         -> bounded Python uploader
                         -> papers.token_usage_logs
```

The optional `pi-logging` extra adds the existing PostgreSQL driver and dotenv configuration pattern. Each project opts in through `<cwd>/.pi/usage-logging.json`; its explicit credential file is authoritative even in tmux children. Existing `run_query()` callbacks and consumer-owned writers are unchanged.

Capture is local-first, one immutable UUID per usage event. Upload removes files only after a successful commit; the table's primary key makes retries duplicate-safe. Database failures retain files, while local capture failure terminates Pi. See the README for installation, retry commands, and coverage limits.

Keep one timeout owner per upload. The Pi extension runs the worker directly under `pi.exec`; manual and startup uploads use Python supervision. Nesting these deadlines can kill the supervisor and leave its database worker running. Check Pi's `killed` flag as well as the exit code, because a terminated command can report code zero.

Send each bounded batch in one parameterized INSERT. Per-record round trips can exceed the upload deadline and repeatedly roll back the same backlog even when the database is healthy.

### `cache.py` — Prompt caching

`add_cache_control()` injects `cache_control: {type: ephemeral}` into a message at a given index, enabling Anthropic's prompt caching. No-ops for non-Claude models.

### `vision.py` — Multi-modal formatting

`format_vision_messages()` orders image/text content blocks per provider convention (Claude: images first; others: text first).

## Design Patterns

- **Query routing state** is limited to the optional usage callback.
- **No env loading in query routing** — consumers handle their own `.env`. The opt-in logging CLI reads only the credential file named in its project profile.
- **Pydantic for structured output** across backends (Instructor for LiteLLM, `output_format` schema for Agent SDK, native Codex schema, prompt-instructed Pi JSON validation).
- **Errors surface naturally** — no blanket try/except. Retries only on transient API failures.

## How to Install (dev)

```bash
cd llm-query-utils
uv sync
```

To use from another project:

```bash
uv add llm-query-utils --git https://github.com/manolorueda/llm-query-utils.git
```
