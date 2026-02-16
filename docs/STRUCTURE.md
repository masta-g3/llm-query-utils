# llm-query-utils — Architecture

## Purpose

Reusable LLM query layer extracted from [llmpedia-workflows](https://github.com/manolorueda/llmpedia-workflows). Provides a single `run_query()` function that transparently routes queries to the appropriate backend based on the model requested.

**Target users:** Internal projects that need LLM calls with structured output, retries, caching, and usage tracking without duplicating plumbing.

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Query routing | `run_query()` | Single entrypoint, auto-selects backend |
| Claude backend | `claude-agent-sdk` | Native structured output, tool use |
| Other models | `litellm` + `instructor` | OpenAI-compatible, structured output via tool-calling |
| Schemas | `pydantic` v2 | Shared between both backends |
| Cost tracking | `tokencost` | Per-call token/cost calculation |
| Packaging | `uv` + `uv_build` | Fast, lockfile-based |

## Architecture

```mermaid
flowchart TD
    A[run_query] -->|Claude model?| B{Agent SDK?}
    B -->|Yes| C[_agent_sdk_query]
    B -->|No: custom messages,\nextended thinking| D[LiteLLM / Instructor]
    A -->|Non-Claude model| D

    C --> E[Usage callback]
    D --> E
    E --> F[Consumer logs/DB]
```

### Routing rules

`run_query()` defaults to Agent SDK for Claude models, but falls back to LiteLLM when:
- Custom `messages` list is provided (multi-turn)
- Extended thinking is enabled
- Model name doesn't contain "claude"

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
    ├── query.py             # run_query() + _agent_sdk_query()
    ├── cache.py             # add_cache_control() for Anthropic prompt caching
    ├── vision.py            # format_vision_messages() for multi-modal
    └── usage.py             # UsageData dataclass + callback mechanism
```

## Key Modules

### `query.py` — Core routing

- **`run_query()`**: Synchronous entrypoint. Routes to Agent SDK or LiteLLM based on model/options. Handles retries (0s, 30s, 60s, 120s), extended thinking setup, and temperature disabling for reasoning models (o1/o3/gpt-5).
- **`_agent_sdk_query()`**: Async Agent SDK handler. Streams messages, counts tool calls, extracts structured output via `output_format`.

### `usage.py` — Usage tracking

Consumer registers a callback via `set_usage_callback(fn)`. Every query fires `UsageData` (tokens, costs, cache stats) to the callback. This decouples the library from any specific logging/DB implementation.

### `cache.py` — Prompt caching

`add_cache_control()` injects `cache_control: {type: ephemeral}` into a message at a given index, enabling Anthropic's prompt caching. No-ops for non-Claude models.

### `vision.py` — Multi-modal formatting

`format_vision_messages()` orders image/text content blocks per provider convention (Claude: images first; others: text first).

## Design Patterns

- **No global state** beyond the optional usage callback.
- **No env loading** — consumers handle their own `.env`.
- **Pydantic for structured output** in both backends (Instructor for LiteLLM, `output_format` schema for Agent SDK).
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
