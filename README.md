# llm-query-utils

Minimal LLM query routing via Claude Agent SDK and LiteLLM/Instructor.

A single `run_query()` entrypoint that routes to the right backend:
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
```

## Modules

| Module | Purpose |
|--------|---------|
| `query.py` | `run_query()` — routing, retries, extended thinking |
| `config.py` | Default model constants |
| `cache.py` | Anthropic prompt caching helpers |
| `vision.py` | Multi-modal message formatting |
| `usage.py` | Token/cost tracking via callback |
