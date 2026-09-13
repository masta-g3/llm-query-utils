# llm-query-utils

Minimal LLM query routing via CLI/SDK transports and LiteLLM/Instructor.

A single `run_query()` entrypoint that routes to the right backend:
- **`use_pi_sdk=True`** → Pi CLI JSON mode
- **`gpt-5*` model + `use_codex_sdk=True`** → Codex CLI (ChatGPT login)
- **Claude models** → Agent SDK (structured output, tool use)
- **Everything else** → LiteLLM/Instructor (OpenAI-compatible)

## Install

```bash
uv add llm-query-utils --git https://github.com/manolorueda/llm-query-utils.git
```

## Usage

```python
from llm_query_utils import run_query

# Plain text query
answer = run_query(
    system_message="You are a helpful assistant.",
    user_message="Explain transformers in one sentence.",
)

# Structured output
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]

result = run_query(
    user_message="Summarize the key ideas of attention mechanisms.",
    model=Summary,
)

# Non-Claude model (routes through LiteLLM)
answer = run_query(
    user_message="Hello",
    llm_model="gpt-5-mini",
)

# ChatGPT subscription provider via LiteLLM OAuth device flow
answer = run_query(
    user_message="Hello",
    llm_model="chatgpt/gpt-5.4",
    use_agent_sdk=False,
)

# Codex route for any gpt-5* model (uses local Codex CLI session; no OPENAI_API_KEY required)
answer = run_query(
    user_message="Reply with exactly: codex_ok",
    llm_model="gpt-5.3-codex",
    use_codex_sdk=True,
    use_agent_sdk=False,
)

# Pi CLI route (uses local Pi configuration/auth)
answer = run_query(
    user_message="Reply with exactly: pi_ok",
    llm_model="anthropic/claude-sonnet-4-5",
    use_pi_sdk=True,
    use_agent_sdk=False,
    pi_options={"timeout": 120},
)

# Pi route with an explicit provider/model/thinking level
answer = run_query(
    system_message="You are concise.",
    user_message="Reply with exactly: pi_gpt_ok",
    llm_model="gpt-5.5",
    use_pi_sdk=True,
    use_agent_sdk=False,
    pi_options={
        "provider": "openai-codex",
        "timeout": 900,
        "thinking": "high",
    },
)
```

## Environment Variables

The underlying SDKs read API keys from the environment — set whichever you need:

| Variable | Required for |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude models (Agent SDK) |
| `OPENAI_API_KEY` | OpenAI API models (LiteLLM, e.g. `gpt-5-mini`) |
| `CHATGPT_TOKEN_DIR` | Optional stable token directory for LiteLLM `chatgpt/...` OAuth |
| `ANTHROPIC_EXTENDED_THINKING_BETA` | Optional — beta header for extended thinking |

ChatGPT subscription access uses LiteLLM's `chatgpt/...` provider and requires `litellm>=1.81.11`. On first use, LiteLLM prompts for an OAuth device-code login; set `CHATGPT_TOKEN_DIR` for scheduled jobs so the same token cache is reused.

Codex route auth is different:
- authenticate once with `codex login`
- verify with `codex login status` (must show `Logged in using ChatGPT`)
- then call `run_query(... use_codex_sdk=True, llm_model="gpt-5.3-codex")`
- pass `codex_options={"timeout": 900}` to fail a hung `codex exec` call clearly
- any model whose basename starts with `gpt-5` is routed through Codex when `use_codex_sdk=True`, including provider-prefixed names like `chatgpt/gpt-5.4`
- legacy alias `gpt-5-codex` is still accepted for compatibility
- usage callbacks include exact Codex token counts plus LiteLLM API-equivalent estimated costs; these are not actual ChatGPT subscription spend

If not logged in, `run_query()` raises:
- `Codex CLI is not authenticated. Run codex login with ChatGPT before using use_codex_sdk=True.`

Pi route auth/config is handled by the local `pi` CLI:
- install/configure Pi separately
- authenticate or set provider API keys as Pi expects
- call `run_query(... use_pi_sdk=True, llm_model="anthropic/claude-sonnet-4-5")`
- set `pi_options={"provider": "openai-codex"}` when a model name like `gpt-5.5` must resolve to Pi's Codex-backed provider instead of another configured provider
- use `pi_options={"thinking": "low"}` or `{"thinking": "high"}` for Pi thinking levels; provider kwargs like `reasoning_effort` are not translated by this package
- by default the Pi route is ephemeral, disables Pi tools, and disables discovered resources for deterministic query behavior
- structured output is requested through prompt instructions and validated locally with Pydantic; use the LiteLLM/Instructor route when native tool-call schemas are required

Supported `pi_options` keys: `model`, `provider`, `thinking`, `tools`, `extensions`, `timeout`, `cwd`, `isolate_resources`, `no_session`.

Load selected extensions without enabling resource discovery:

```python
pi_options={
    "provider": "openai-codex",
    "extensions": ["/absolute/path/to/pi-web-access/index.ts"],
    "tools": ["web_search", "get_search_content"],
    "timeout": 900,
}
```

`extensions` accepts a list of non-empty Pi extension sources and adds explicit `--extension` flags. Resource isolation remains enabled by default. `tools` allowlists both built-in and extension tools; omitting it keeps all tools disabled. Extensions execute trusted local code, so load only reviewed sources.

Pi callers can opt in to the typed result and usage pair with `return_usage=True`:

```python
from llm_query_utils import QueryRun

run = run_query(..., use_pi_sdk=True, return_usage=True)
assert isinstance(run, QueryRun)
print(run.result, run.usage.total_cost if run.usage else None)
```

Other routes reject `return_usage=True`. Default return values do not change. `UsageData.total_cost` is nullable. It is the provider whole-call estimate when known and includes cache cost once.

## PostgreSQL usage callback

Apply `migrations/001_central_usage.sql` before releasing these writers. See [central usage logging](docs/USAGE_LOGGING.md) for the table contract, verified totals, historical migration, and guarded rollout. No query or uploader changes the schema automatically.

Install the `pi-logging` extra for PostgreSQL support. Register one callback with an explicit project and a fresh DBAPI connection factory:

```python
from llm_query_utils import postgres_usage_callback, set_usage_callback

set_usage_callback(postgres_usage_callback(
    project_id="sentiment",
    connect=get_connection,
))
```

The callback skips events without `process_id`. For other events, it opens one connection, inserts into `papers.token_usage_logs`, commits once, and closes the connection. Database errors surface. `insert_usage_rows(connection, rows)` is also available from `llm_query_utils.usage_db` for caller-owned batches; each tuple must follow `USAGE_COLUMNS`, and the function does not commit or close the connection.

## Logging direct Pi agents

`pi-logged` starts Pi with an explicit usage-logging extension. It preserves Pi arguments, tools, model choices, terminal output, and process signals. Use it for direct Pi agents, not as a replacement for the `run_query()` transport. Existing Python usage callbacks remain unchanged.

Install the optional PostgreSQL support and console commands:

```bash
uv tool install --editable '/Users/manager/Code/llm-query-utils[pi-logging]'
```

In the calling project's root, create `.pi/usage-logging.json`:

```json
{
  "project_id": "llmpedia",
  "env_file": "../.env",
  "table": "papers.token_usage_logs"
}
```

`env_file` resolves relative to this profile. Its `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_HOST`, and `DB_PORT` values are authoritative for logging; stale environment values from a tmux server do not override them. Use literal credential values; the logger does not expand `${...}` references. Keep credentials out of the profile and version control. The logger targets the existing `papers.token_usage_logs` schema; it does not create tables.

Run from that project root:

```bash
pi-logged --print 'Reply with OK'
```

The launcher resolves normal `pi` from PATH and explicitly loads the packaged extension, including when `--no-extensions` is present. It also points `PI_TMUX_SUBAGENTS_PI_BIN` at itself, so tmux children use the same launcher. Each child loads the project profile independently. Do not alias normal `pi` to this command or auto-load the extension globally: existing Python callbacks could otherwise log the same calls twice.

Reported response and default-compaction usage is saved first in private local files under `~/.local/state/llm-query-utils/pi/<project_id>/`. Each record has a UUID reused for every upload attempt. PostgreSQL inserts use `ON CONFLICT (id) DO NOTHING`; files are removed only after commit. Destination fingerprints prevent pending records from being redirected by a changed profile.

Database failures leave records pending and do not stop Pi. Startup, run completion, compaction, and shutdown make bounded upload attempts. Retry manually with:

```bash
llm-usage flush --config /absolute/project/.pi/usage-logging.json
```

Manual flush reports failure when delivery errors remain. A local write failure is different: the extension exits Pi with code 74 rather than continue without capture. The logger never stores prompts, answers, tools, or credentials.

Response model labels prefer Pi's provider-reported `responseModel`; default compaction records use the active catalog model and may aggregate two summary calls. Missing usage, custom summarizers without attribution, and hard kills before capture remain coverage gaps. Costs are API-equivalent estimates, not subscription charges. Unknown costs remain unknown, not invented zeroes. Cache counts and costs stay separate from uncached input.

## Modules

| Module | Purpose |
|--------|---------|
| `query.py` | `run_query()` — Pi/Codex/Agent/LiteLLM routing, retries, extended thinking |
| `config.py` | Default model constants |
| `cache.py` | Anthropic prompt caching helpers |
| `vision.py` | Multi-modal message formatting |
| `usage.py` | Shared usage metadata, complete Pi totals, and callback dispatch |
| `usage_db.py` | Project-attributed PostgreSQL callback and shared batch INSERT |
| `pi_logging.py` | Opt-in `pi-logged` launcher and duplicate-safe PostgreSQL upload |
| `pi_usage_extension.ts` | Packaged Pi event capture with durable local pending records |
