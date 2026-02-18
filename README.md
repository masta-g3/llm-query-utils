# llm-query-utils

Minimal LLM query routing via Claude Agent SDK and LiteLLM/Instructor.

A single `run_query()` entrypoint that routes to the right backend:
- **Codex model (`gpt-5-codex`) + `use_codex_sdk=True`** -> Codex CLI (ChatGPT login)
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

# Codex route (uses local Codex CLI session; no OPENAI_API_KEY required)
answer = run_query(
    user_message="Reply with exactly: codex_ok",
    llm_model="gpt-5-codex",
    use_codex_sdk=True,
    use_agent_sdk=False,
)
```

## Environment Variables

The underlying SDKs read API keys from the environment — set whichever you need:

| Variable | Required for |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude models (Agent SDK) |
| `OPENAI_API_KEY` | OpenAI models (LiteLLM) |
| `ANTHROPIC_EXTENDED_THINKING_BETA` | Optional — beta header for extended thinking |

Codex route auth is different:
- authenticate once with `codex login`
- verify with `codex login status` (must show `Logged in using ChatGPT`)
- then call `run_query(... use_codex_sdk=True, llm_model="gpt-5-codex")`

If not logged in, `run_query()` raises:
- `Codex CLI is not authenticated. Run codex login with ChatGPT before using use_codex_sdk=True.`

## Modules

| Module | Purpose |
|--------|---------|
| `query.py` | `run_query()` — Codex/Agent/LiteLLM routing, retries, extended thinking |
| `config.py` | Default model constants |
| `cache.py` | Anthropic prompt caching helpers |
| `vision.py` | Multi-modal message formatting |
| `usage.py` | Token/cost tracking via callback (LiteLLM pricing on LiteLLM route) |
