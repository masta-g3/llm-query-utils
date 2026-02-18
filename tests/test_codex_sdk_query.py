import asyncio

from pydantic import BaseModel

import llm_query_utils.query as query_module
from llm_query_utils.usage import set_usage_callback


class OutputSchema(BaseModel):
    status: str
    count: int


def test_codex_query_parses_structured_output_and_fires_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    monkeypatch.setattr(query_module, "_run_codex_login_preflight", lambda: None)
    monkeypatch.setattr(
        query_module,
        "_run_codex_exec",
        lambda model, prompt, output_schema, extra_args: (
            '{"status":"ok","count":7}',
            {"input_tokens": 12, "output_tokens": 4, "cached_input_tokens": 2},
            "",
        ),
    )

    result = asyncio.run(
        query_module._codex_sdk_query(
            system_message="be precise",
            user_message="return json",
            llm_model="gpt-5-codex",
            output_schema=OutputSchema,
            process_id="p-1",
        )
    )

    assert isinstance(result, OutputSchema)
    assert result.count == 7
    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 12
    assert usage_events[0].completion_tokens == 4
    assert usage_events[0].cache_read_input_tokens == 2


def test_codex_query_returns_text(monkeypatch):
    monkeypatch.setattr(query_module, "_run_codex_login_preflight", lambda: None)
    monkeypatch.setattr(
        query_module,
        "_run_codex_exec",
        lambda model, prompt, output_schema, extra_args: (
            "plain-text",
            {"input_tokens": 3, "output_tokens": 1},
            "",
        ),
    )

    result = asyncio.run(
        query_module._codex_sdk_query(
            system_message=None,
            user_message="hello",
            llm_model="gpt-5-codex",
            output_schema=None,
        )
    )

    assert result == "plain-text"


def test_codex_options_extra_args_must_be_list():
    try:
        asyncio.run(
            query_module._codex_sdk_query(
                system_message=None,
                user_message="hello",
                llm_model="gpt-5-codex",
                codex_options={"extra_args": "--bad"},
            )
        )
    except ValueError as exc:
        assert "codex_options.extra_args must be a list" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-list codex_options.extra_args")


def test_codex_options_model_must_be_supported():
    try:
        asyncio.run(
            query_module._codex_sdk_query(
                system_message=None,
                user_message="hello",
                llm_model="gpt-5-codex",
                codex_options={"model": "codex-mini-latest"},
            )
        )
    except ValueError as exc:
        assert "Unsupported Codex model" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported codex_options.model")
