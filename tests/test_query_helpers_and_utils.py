import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import llm_query_utils.query as query_module
from llm_query_utils.cache import add_cache_control
from llm_query_utils.vision import format_vision_messages
from llm_query_utils.usage import set_usage_callback


class OutputSchema(BaseModel):
    status: str
    count: int


def test_agent_sdk_query_raises_on_error_and_empty_response(monkeypatch):
    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def error_stream(*args, **kwargs):
        yield FakeResultMessage(is_error=True, result="bad request")

    monkeypatch.setattr(query_module, "agent_sdk_query", error_stream)

    with pytest.raises(RuntimeError, match="Agent SDK error"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )

    async def empty_stream(*args, **kwargs):
        yield FakeResultMessage(is_error=False, result=None)

    monkeypatch.setattr(query_module, "agent_sdk_query", empty_stream)

    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )


def test_agent_sdk_query_emits_zero_usage_on_terminal_success_without_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def success_stream(*args, **kwargs):
        yield FakeAssistantMessage([FakeTextBlock("agent-ok")])
        yield FakeResultMessage(usage=None, total_cost_usd=0.42, is_error=False)

    monkeypatch.setattr(query_module, "agent_sdk_query", success_stream)

    result = asyncio.run(
        query_module._agent_sdk_query(
            system_message="system",
            user_message="hello",
            llm_model="claude-3-5-sonnet",
            process_id="p-1",
        )
    )

    assert result == "agent-ok"
    assert len(usage_events) == 1
    assert usage_events[0].process_id == "p-1"
    assert usage_events[0].prompt_tokens == 0
    assert usage_events[0].completion_tokens == 0
    assert usage_events[0].prompt_cost == 0.42
    assert usage_events[0].completion_cost == 0


def test_agent_sdk_query_emits_usage_with_present_payload(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def success_stream(*args, **kwargs):
        yield FakeAssistantMessage([FakeTextBlock("agent-ok")])
        yield FakeResultMessage(
            usage={
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 1,
            },
            total_cost_usd=0.77,
            is_error=False,
        )

    monkeypatch.setattr(query_module, "agent_sdk_query", success_stream)

    result = asyncio.run(
        query_module._agent_sdk_query(
            system_message="system",
            user_message="hello",
            llm_model="claude-3-5-sonnet",
            process_id="p-usage",
        )
    )

    assert result == "agent-ok"
    assert len(usage_events) == 1
    assert usage_events[0].process_id == "p-usage"
    assert usage_events[0].prompt_tokens == 12
    assert usage_events[0].completion_tokens == 5
    assert usage_events[0].cache_creation_input_tokens == 3
    assert usage_events[0].cache_read_input_tokens == 1
    assert usage_events[0].prompt_cost == 0.77
    assert usage_events[0].completion_cost == 0


def test_agent_sdk_query_emits_usage_before_terminal_error(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def error_stream(*args, **kwargs):
        yield FakeResultMessage(usage=None, total_cost_usd=0.0, is_error=True, result="bad request")

    monkeypatch.setattr(query_module, "agent_sdk_query", error_stream)

    with pytest.raises(RuntimeError, match="Agent SDK error"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )

    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 0
    assert usage_events[0].completion_tokens == 0


def test_agent_sdk_query_transport_error_before_terminal_emits_no_usage(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    class FakeResultMessage:
        def __init__(
            self,
            *,
            usage=None,
            total_cost_usd=0,
            is_error=False,
            result=None,
            structured_output=None,
        ):
            self.usage = usage
            self.total_cost_usd = total_cost_usd
            self.is_error = is_error
            self.result = result
            self.structured_output = structured_output

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeToolUseBlock:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(query_module, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(query_module, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(query_module, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(query_module, "ToolUseBlock", FakeToolUseBlock)

    async def broken_stream(*args, **kwargs):
        raise RuntimeError("stream failure")
        yield  # pragma: no cover

    monkeypatch.setattr(query_module, "agent_sdk_query", broken_stream)

    with pytest.raises(RuntimeError, match="stream failure"):
        asyncio.run(
            query_module._agent_sdk_query(
                system_message="system",
                user_message="hello",
                llm_model="claude-3-5-sonnet",
            )
        )

    assert usage_events == []


def test_fire_agent_usage_handles_malformed_payload():
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    query_module._fire_agent_usage(
        llm_model="claude-3-5-sonnet",
        process_id="p-malformed",
        usage_data="not-a-dict",
        total_cost_usd=None,
    )

    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 0
    assert usage_events[0].completion_tokens == 0
    assert usage_events[0].cache_creation_input_tokens == 0
    assert usage_events[0].cache_read_input_tokens == 0
    assert usage_events[0].prompt_cost == 0


def test_parse_codex_event_stream_extracts_usage_and_error():
    event_stream = "\n".join(
        [
            "noise",
            '{"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":3}}',
            '{"type":"turn.failed","error":{"message":"turn failed"}}',
            '{"type":"error","message":"runtime failure"}',
        ]
    )

    usage, error = query_module._parse_codex_event_stream(event_stream)

    assert usage == {"input_tokens": 11, "output_tokens": 3}
    assert error == "runtime failure"


def test_fire_codex_usage_accepts_both_cache_read_token_keys():
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))

    query_module._fire_codex_usage(
        {"input_tokens": 2, "output_tokens": 1, "cache_read_input_tokens": 9},
        llm_model="gpt-5-codex",
        process_id="p-1",
    )
    query_module._fire_codex_usage(
        {"input_tokens": 2, "output_tokens": 1, "cached_input_tokens": 7},
        llm_model="gpt-5-codex",
        process_id="p-2",
    )

    assert usage_events[0].cache_read_input_tokens == 9
    assert usage_events[1].cache_read_input_tokens == 7


def test_add_cache_control_only_for_supported_models_and_valid_index():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "prompt"},
    ]

    updated = add_cache_control(messages, cache_message_index=1, llm_model="claude-3-5-sonnet")
    assert updated[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[1]

    invalid_index = add_cache_control(messages, cache_message_index=9, llm_model="claude-3-5-sonnet")
    assert invalid_index == messages

    unsupported_model = add_cache_control(messages, cache_message_index=0, llm_model="gpt-5-mini")
    assert unsupported_model == messages


def test_format_vision_messages_applies_provider_order_and_url_formatting():
    images = ["BASE64_IMAGE", "https://example.com/image.png"]

    claude_messages = format_vision_messages(
        images=images,
        text="describe this",
        model="claude-3-5-sonnet",
        system_message="vision rules",
    )
    claude_content = claude_messages[1]["content"]

    assert claude_messages[0]["role"] == "system"
    assert claude_content[0]["type"] == "image_url"
    assert claude_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert claude_content[1]["image_url"]["url"] == "https://example.com/image.png"
    assert claude_content[-1] == {"type": "text", "text": "describe this"}

    other_messages = format_vision_messages(
        images=images,
        text="describe this",
        model="gpt-5-mini",
    )
    other_content = other_messages[0]["content"]

    assert other_content[0] == {"type": "text", "text": "describe this"}
    assert other_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert other_content[2]["image_url"]["url"] == "https://example.com/image.png"


def test_log_usage_emits_costs_for_gpt5_codex(monkeypatch):
    usage_events = []
    set_usage_callback(lambda data: usage_events.append(data))
    monkeypatch.setattr(query_module, "cost_per_token", lambda **kwargs: (1.25, 2.5))
    monkeypatch.setattr(
        query_module,
        "get_model_info",
        lambda **kwargs: {
            "input_cost_per_token": 0.5,
            "cache_creation_input_token_cost": None,
            "cache_read_input_token_cost": 0.1,
        },
    )

    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    query_module._log_usage(usage, llm_model="gpt-5-codex")

    assert len(usage_events) == 1
    assert usage_events[0].prompt_cost > 0
    assert usage_events[0].completion_cost > 0


def test_calculate_usage_costs_falls_back_for_cache_creation_rate(monkeypatch):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        cache_creation_input_tokens=4,
        cache_read_input_tokens=3,
    )

    monkeypatch.setattr(query_module, "cost_per_token", lambda **kwargs: (1.25, 2.5))
    monkeypatch.setattr(
        query_module,
        "get_model_info",
        lambda **kwargs: {
            "input_cost_per_token": 0.5,
            "cache_creation_input_token_cost": None,
            "cache_read_input_token_cost": 0.1,
        },
    )

    prompt_cost, completion_cost, cache_creation_cost, cache_read_cost = (
        query_module._calculate_usage_costs(usage, "gpt-5-codex")
    )

    assert prompt_cost == 1.25
    assert completion_cost == 2.5
    assert cache_creation_cost == 4.0
    assert cache_read_cost == 0.30000000000000004


def test_calculate_usage_costs_matches_litellm_totals_for_supported_model():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    expected_prompt_cost, expected_completion_cost = query_module.cost_per_token(
        model="gpt-4o-mini",
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
    )

    prompt_cost, completion_cost, _, _ = query_module._calculate_usage_costs(
        usage, "gpt-4o-mini"
    )

    assert prompt_cost == expected_prompt_cost
    assert completion_cost == expected_completion_cost


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
    with pytest.raises(ValueError, match="codex_options.extra_args must be a list"):
        asyncio.run(
            query_module._codex_sdk_query(
                system_message=None,
                user_message="hello",
                llm_model="gpt-5-codex",
                codex_options={"extra_args": "--bad"},
            )
        )


def test_codex_options_model_must_be_supported():
    with pytest.raises(ValueError, match="Unsupported Codex model"):
        asyncio.run(
            query_module._codex_sdk_query(
                system_message=None,
                user_message="hello",
                llm_model="gpt-5-codex",
                codex_options={"model": "codex-mini-latest"},
            )
        )
