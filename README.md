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

Supported `pi_options` keys: `model`, `provider`, `thinking`, `tools`, `timeout`, `cwd`, `isolate_resources`, `no_session`.

## Modules

| Module | Purpose |
|--------|---------|
| `query.py` | `run_query()` — Pi/Codex/Agent/LiteLLM routing, retries, extended thinking |
| `config.py` | Default model constants |
| `cache.py` | Anthropic prompt caching helpers |
| `vision.py` | Multi-modal message formatting |
| `usage.py` | Token/cost tracking via callback; Codex costs are LiteLLM API-equivalent estimates, not subscription spend |
